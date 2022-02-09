import logging
from typing import Text, List
from src.utils.io import read_json
from src.utils.fuzzy import is_relevant_string
from src.utils.common import is_float
from src.search_engine import ESKnowLife
from src.utils.constants import (
    MEDICAL_TEST_PATH, 
    QUANTITATIVE_PATH, 
    POSITIVE_TEXT,
    TestResult
)

logger = logging.getLogger(__name__)
class MedicalTest():
    def __init__(self, medical_test_path=MEDICAL_TEST_PATH, quantitative_path= QUANTITATIVE_PATH) -> None:
        self.medical_test = read_json(medical_test_path)
        self.quantitative = read_json(quantitative_path)
        logger.info("Medical test loaded")

    def get_suggestions(self, indicators: List):
        suggestions = []
        count = 0
        for i in indicators:
            count += 1
            sg = {
                "id": count,
                "input": i,
            }
            references = []
            for m in self.medical_test:
                if is_relevant_string(i["test_name"], m["name"], method=['exact','fuzzy'], score=90):
                    sg["name"] = m["name"]
                    sg["overview"] = m["overview"]
                    if m["references"]:
                        references.extend(m["references"])
                    for q in self.quantitative:
                        if q["medical_test"]["id"] == m["id"]:
                            if q["test_result"] == TestResult.positive.value and is_relevant_string(str(i["result"]), POSITIVE_TEXT, score=90, remove_accent=True):
                               sg["note"] = q["note"]
                               sg["cause"] = q["cause"]
                               sg["recommend"] = q["recommend"]
                               if m["references"]:
                                   references.extend(m["references"])
                               break
                            elif is_float(str(i["result"])):
                                test_result = float(str(i["result"]))
                                if test_result >= q["min_value"] and  test_result <= q["max_value"]:
                                    sg["note"] = q["note"]
                                    sg["cause"] = q["cause"]
                                    sg["recommend"] = q["recommend"]
                                    if m["references"]:
                                        references.extend(m["references"])
                                    break
                    break
            sg["references"] = list(dict.fromkeys(references))
            suggestions.append(sg)
        return suggestions

        