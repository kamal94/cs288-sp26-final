import pandas as pd
import os

file_path = os.path.join('..', 'outputs', 'static_eval.csv')

try:
    df = pd.read_csv(file_path, header=None)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

def analyze_dynamic_turns(df):
    turn_indices = list(df.columns[2:])
    num_transitions = len(turn_indices) - 1

    def calculate_row_metrics(row):
        actual_changes = 0
        if num_transitions > 0:
            for i in range(num_transitions):
                if row[turn_indices[i]] != row[turn_indices[i+1]]:
                    actual_changes += 1
            stability = 1 - (actual_changes / num_transitions)
        else:
            stability = 1.0

        is_confident = 1 if row[turn_indices[0]] == row[turn_indices[-1]] else 0

        t_first_correct = (row[turn_indices[0]] == row[0])
        t_last_correct = (row[turn_indices[-1]] == row[0])
        
        direction = "Stayed"
        if t_first_correct and not t_last_correct:
            direction = "Right_to_Wrong"
        elif not t_first_correct and t_last_correct:
            direction = "Wrong_to_Right"
        elif t_first_correct and t_last_correct:
            direction = "Stayed_Right"
        else:
            direction = "Stayed_Wrong"
            
        return pd.Series([stability, is_confident, direction], 
                         index=['stability', 'confidence', 'direction'])

    results_df = df.apply(calculate_row_metrics, axis=1)
    combined_df = pd.concat([df, results_df], axis=1)

    summary = combined_df.groupby(1).agg(
        avg_stability=('stability', 'mean'),
        confidence_pct=('confidence', lambda x: x.mean() * 100),
        sample_size=(1, 'count')
    ).sort_values('confidence_pct')
    
    summary.index.name = 'static_type'

    directional = combined_df.groupby([1, 'direction']).size().unstack(fill_value=0)
    directional.index.name = 'static_type'

    return summary, directional

summary_stats, directional_stats = analyze_dynamic_turns(df)

print("Efficacy and Stability Summary: ")
print(summary_stats)
print("\nDirectional Outcomes: ")
print(directional_stats)