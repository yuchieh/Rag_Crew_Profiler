#!/usr/bin/env python
import sys
import json
import warnings

from first_crew.crew_collaborative import CollaborativeCrew
from first_crew.utils import extract_json_from_output

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the collaborative crew using the first testing sample.
    """
    test_json_path = "data/test_review_subset.json"
    with open(test_json_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        test_case = json.loads(first_line)

    inputs = {
        'user_id': test_case['user_id'],
        'item_id': test_case['item_id']
    }

    print(f"Starting Collaborative Prediction for User: {inputs['user_id']} | Item: {inputs['item_id']}")

    try:
        result = CollaborativeCrew().crew().kickoff(inputs=inputs)
        
        # Parse and sanitize the LLM output into clean JSON
        report = extract_json_from_output(result.raw)
        
        # Write clean report
        with open('report_collaborative.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Collaborative Prediction Completed ===")
        print(f"Stars: {report.get('stars')}")
        print(f"Review: {report.get('review', '')[:100]}...")
        print(f"Results written to report_collaborative.json")
        
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
