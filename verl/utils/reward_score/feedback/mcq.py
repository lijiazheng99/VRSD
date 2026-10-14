import re


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def is_correct_format(text: str) -> bool:
    """
    Check if the text is in the correct XML format.

    The text should contain at the end of the text:
    <answer>
    (A|B|C|D)
    </answer>
    """
    # Closed tag at the end, OR an unclosed "<answer>\nC<eos>" — OLMo-3-7B-Instruct emits EOS right
    # after the letter. extract_xml_answer already scores that form correctly; this only keeps the format flag honest.
    pattern = r"<answer>\s*(A|B|C|D)\s*(</answer>)?\s*$"
    return re.search(pattern, text.rstrip()) is not None

def compute_score(solution: str, ground_truth: str) -> dict:
    multiple_choice_answer = extract_xml_answer(solution)

    reward = float(multiple_choice_answer == ground_truth)
    incorrect_format = not is_correct_format(solution)   # was inverted (flag=1 when the format WAS correct); logging-only field

    return {
      "score": reward,
      "acc": reward,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if incorrect_format else 0,
      "feedback": "",
    }
