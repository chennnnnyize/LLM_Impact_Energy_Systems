#!/usr/bin/env python3
import re
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Enhanced dataset processing with improved prompts and response handling"""
    
    # Language mapping for translation tasks
    LANGUAGE_NAMES = {
        'ar': 'Arabic',
        'en': 'English',
        'de': 'German',
        'fr': 'French',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ru': 'Russian',
        'es': 'Spanish',
        'pt': 'Portuguese'
    }

    @staticmethod
    def get_language_name(code: str) -> str:
        """Convert language code to full name."""
        return DatasetProcessor.LANGUAGE_NAMES.get(code.lower(), code)

    @staticmethod
    def validate_dataset_example(dataset_name: str, example: Dict[str, Any]) -> Optional[str]:
        """Validate dataset example format and required fields."""
        try:
            # Common validation for content/label datasets
            if dataset_name in ["sst2", "bool_logic", "valid_parentheses"]:
                if 'content' not in example:
                    return f"Missing 'content' field in {dataset_name} example"
                if 'label' not in example:
                    return f"Missing 'label' field in {dataset_name} example"
                
            # MMLU specific validation
            elif dataset_name == "mmlu":
                required = ['question', 'subject', 'choices', 'answer']
                missing = [f for f in required if f not in example]
                if missing:
                    return f"Missing required fields in MMLU example: {missing}"
                if len(example['choices']) != 4:
                    return "MMLU example must have exactly 4 choices"
                if not isinstance(example['answer'], int) or not (0 <= example['answer'] <= 3):
                    return "MMLU answer must be an integer between 0 and 3"
                
            # Math specific validation
            elif dataset_name.startswith("math"):
                if 'question' not in example:
                    return "Missing 'question' field in math example"
                if 'answer' not in example:
                    return "Missing 'answer' field in math example"
                
            # UN Multi specific validation
            elif dataset_name == "un_multi":
                if 'source' not in example:
                    return "Missing 'source' field in translation example"
                if 'target' not in example:
                    return "Missing 'target' field in translation example"
                if 'soruce_lang' not in example or 'target_lang' not in example:  # Note: keeping typo
                    return "Missing language fields in translation example"
                
            # IWSLT2017 specific validation
            elif dataset_name == "iwslt2017":
                if 'source' not in example:
                    return "Missing 'source' field in translation example"
                if 'target' not in example:
                    return "Missing 'target' field in translation example"
                
            return None
        except Exception as e:
            return f"Error validating example: {str(e)}"

    @staticmethod
    def format_prompt(dataset_name: str, example: Dict[str, Any]) -> str:
        """Format prompts with improved structure and clarity for each dataset type."""
        try:
            # SST2 prompt
            if dataset_name == "sst2":
                text = example.get('content', '')
                return f"""[INST] Sentiment Analysis Task:
Classify the following text as POSITIVE or NEGATIVE.

Text: "{text}"

Instructions:
- Respond with exactly ONE word
- Must be either POSITIVE or NEGATIVE
- No explanation or additional text allowed

Classification: """

            # Boolean Logic and Valid Parentheses
            elif dataset_name in ["bool_logic", "valid_parentheses"]:
                text = example.get('content', '')
                if dataset_name == "bool_logic":
                    return f"""[INST] Boolean Logic:
Evaluate this expression: {text}

Instructions:
- Respond with exactly TRUE or FALSE
- No explanation needed

Result: """
                else:
                    return f"""[INST] Parentheses Validation:
Expression: {text}

Instructions:
- Check if parentheses are balanced and properly nested
- Consider (), [], and {{}} as valid pairs
- Respond with exactly VALID or INVALID
- No explanation needed

Status: """

            # MMLU prompt
            elif dataset_name == "mmlu":
                question = example.get('question', '')
                subject = example.get('subject', 'General Knowledge')
                choices = example.get('choices', [])
                
                return f"""[INST] Multiple Choice Question:

Subject: {subject.title()}
Question: {question}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Instructions:
- Respond with exactly ONE letter
- Must be A, B, C, or D
- No explanation needed

Answer: """

            # Math prompt
            elif dataset_name.startswith("math"):
                question = example.get('question', '')
                # Clean byte string markers if present
                question = question.replace("b'", "").replace("\\n'", "")
                
                return f"""[INST] Mathematical Problem:

Problem: {question}

Solution Requirements:
1. Show your step-by-step solution
2. Explain each step clearly
3. End with "Final Answer: [numerical_result]"

Solution: """

            # Translation tasks
            elif dataset_name in ["un_multi", "iwslt2017"]:
                source_text = example.get('source', '')
                
                if dataset_name == "un_multi":
                    source_lang = example.get('soruce_lang', 'Source')  # Note: keeping typo
                    target_lang = example.get('target_lang', 'Target')
                    style_note = "Use formal UN-style language"
                else:  # iwslt2017
                    source_lang = "Source"
                    target_lang = "Target"
                    style_note = "Use natural, conversational language"
                
                return f"""[INST] Translation Task:

Source Language: {source_lang}
Target Language: {target_lang}

Source Text:
{source_text}

Requirements:
- Maintain original meaning and tone
- {style_note}
- Preserve technical terms and proper nouns
- Ensure grammatical accuracy

Translation: """

            else:
                logger.warning(f"No specific prompt format for dataset {dataset_name}. Using generic format.")
                return f"[INST] {str(example.get('text', example.get('prompt', str(example))))} "

        except Exception as e:
            logger.error(f"Error formatting prompt for {dataset_name}: {str(e)}")
            raise

    @staticmethod
    def process_response(dataset_name: str, response: str, example: Dict[str, Any]) -> Tuple[Any, bool]:
        """Process model responses with improved validation and cleaning."""
        try:
            # Common preprocessing
            response = response.strip()
            
            # Remove common artifacts including [INST] and [/INST] tags
            response = re.sub(r'\[/?INST\]', '', response)
            response = re.sub(
                r'^(the answer is|answer:|response:|result:|classification:|status:|translation:)\s*',
                '',
                response,
                flags=re.IGNORECASE
            )
            response = response.strip()

            # SST2 processing
            if dataset_name == "sst2":
                response_upper = response.upper()
                # First try exact match
                if response_upper == "POSITIVE":
                    return 1, True
                elif response_upper == "NEGATIVE":
                    return 0, True
                
                # More lenient matching with word boundary check
                response_lower = response.lower()
                if re.search(r'\b(positive|good|great|excellent)\b', response_lower):
                    logger.debug(f"Recovered positive sentiment from: {response}")
                    return 1, True
                elif re.search(r'\b(negative|bad|poor|terrible)\b', response_lower):
                    logger.debug(f"Recovered negative sentiment from: {response}")
                    return 0, True
                
                return response, False

            # Bool Logic and Valid Parentheses processing
            elif dataset_name in ["bool_logic", "valid_parentheses"]:
                response_upper = response.upper()
                
                if dataset_name == "bool_logic":
                    if response_upper == "TRUE":
                        return True, True
                    elif response_upper == "FALSE":
                        return False, True
                    
                    # More lenient matching
                    response_lower = response.lower()
                    if re.search(r'\btrue\b', response_lower) and not re.search(r'\bfalse\b', response_lower):
                        return True, True
                    elif re.search(r'\bfalse\b', response_lower):
                        return False, True
                else:  # valid_parentheses
                    if response_upper == "VALID":
                        return "valid", True
                    elif response_upper == "INVALID":
                        return "invalid", True
                    
                    # More lenient matching
                    response_lower = response.lower()
                    valid_patterns = [r'\bvalid\b', r'\bbalanced\b', r'\bcorrect\b']
                    invalid_patterns = [r'\binvalid\b', r'\bunbalanced\b', r'\bincorrect\b']
                    
                    if any(re.search(pattern, response_lower) for pattern in valid_patterns) and \
                       not any(re.search(pattern, response_lower) for pattern in invalid_patterns):
                        return "valid", True
                    elif any(re.search(pattern, response_lower) for pattern in invalid_patterns):
                        return "invalid", True
                
                return response, False

            # MMLU processing
            elif dataset_name == "mmlu":
                # Remove common artifacts
                response = re.sub(r'\s*\([^)]*\)', '', response)  # Remove parenthetical content
                response = re.sub(r'option\s+', '', response.lower())  # Remove "option" prefix
                
                # First try exact single letter match
                match = re.search(r'\b[ABCD]\b', response.upper())
                if match:
                    letter = match.group(0)
                    return {'A': 0, 'B': 1, 'C': 2, 'D': 3}[letter], True

                # Check for full option text match
                response_lower = response.lower()
                choices_lower = [choice.lower() for choice in example['choices']]
                for i, choice in enumerate(choices_lower):
                    if choice in response_lower:
                        logger.debug(f"Matched answer text for choice {i}")
                        return i, True
                
                return response, False

            # Math processing
            elif dataset_name.startswith("math"):
                # Clean byte string markers if present
                expected_answer = example['answer'].replace("b'", "").replace("\\n'", "")
                
                # Look for explicit "Final Answer:" format
                final_answer_match = re.search(
                    r'final\s*answer\s*:\s*(-?\d*\.?\d+)',
                    response.lower()
                )
                if final_answer_match:
                    try:
                        result = float(final_answer_match.group(1))
                        expected = float(expected_answer)
                        # Allow small floating point differences
                        if abs(result - expected) < 1e-6:
                            return result, True
                    except ValueError:
                        pass
                
                # Look for the last number in the response
                all_numbers = re.findall(r'-?\d*\.?\d+', response)
                if all_numbers:
                    try:
                        result = float(all_numbers[-1])
                        expected = float(expected_answer)
                        if abs(result - expected) < 1e-6:
                            return result, True
                    except ValueError:
                        pass
                
                return response, False

            # Translation processing (UN Multi and IWSLT2017)
            elif dataset_name in ["un_multi", "iwslt2017"]:
                # Clean translation response
                response = response.strip()
                
                # Remove any source text that might have been repeated
                if example.get('source'):
                    response = response.replace(example['source'], '').strip()
                
                # Clean up common artifacts
                response = re.sub(r'^["\']|["\']$', '', response)  # Remove quotes
                response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
                
                # Verify non-empty content
                if response.strip():
                    return response.strip(), True
                return "Invalid empty translation", False

            else:
                logger.warning(f"No specific response processing for dataset {dataset_name}")
                return response.strip(), True

        except Exception as e:
            logger.error(f"Error processing response for {dataset_name}: {str(e)}")
            logger.debug(f"Problematic response: {response}")
            return response, False