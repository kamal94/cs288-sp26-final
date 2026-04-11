import pandas as pd
import os

pd.options.display.float_format = '{:.1f}'.format
pd.options.display.max_columns = None
pd.options.display.width = None

file_path = os.path.join('outputs', 'static_eval_llama-3.1-70b-instruct_20260409_225422.csv')

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    exit()

def analyze_dynamic_turns(df):
    turn_cols = [c for c in df.columns if c.startswith('turn_')]
    num_transitions = len(turn_cols) - 1

    def calculate_row_metrics(row):
        correct = row['correct_answer']
        actual_changes = rw = wr = ww = sr = 0
        # recov_rate: started wrong, probe was correct answer, flipped to right
        wrong_probe_correct_flipped = wrong_probe_correct_total = 0
        # mislead_rate: started right, probe was wrong answer, flipped to wrong
        right_probe_wrong_flipped = right_probe_wrong_total = 0

        for i in range(num_transitions):
            a, b = row[turn_cols[i]], row[turn_cols[i+1]]
            if a != b:
                actual_changes += 1
            a_right, b_right = (a == correct), (b == correct)
            if a_right and not b_right:
                rw += 1
            elif not a_right and b_right:
                wr += 1
            elif not a_right and not b_right:
                ww += 1
            else:
                sr += 1

            probe = row.get(f'turn_{i+2}_probe', '')
            if not a_right and probe == correct:
                wrong_probe_correct_total += 1
                if b_right:
                    wrong_probe_correct_flipped += 1
            if a_right and probe and probe != correct:
                right_probe_wrong_total += 1
                if not b_right:
                    right_probe_wrong_flipped += 1

        stability = 1 - (actual_changes / num_transitions)
        first, last = row[turn_cols[0]], row[turn_cols[-1]]
        fr, lr = first == correct, last == correct

        recov_rate = wrong_probe_correct_flipped / wrong_probe_correct_total if wrong_probe_correct_total > 0 else float('nan')
        mislead_rate = right_probe_wrong_flipped / right_probe_wrong_total if right_probe_wrong_total > 0 else float('nan')

        return pd.Series(
            [stability, sr/num_transitions, rw/num_transitions, wr/num_transitions, ww/num_transitions,
             int(fr and lr), int(fr and not lr), int(not fr and lr), int(not fr and not lr),
             recov_rate, mislead_rate],
            index=['stability', 'Stayed_Right', 'Right_to_Wrong', 'Wrong_to_Right', 'Stayed_Wrong',
                   'rateRR', 'rateRW', 'rateWR', 'rateWW',
                   'recov_rate', 'mislead_rate']
        )

    results_df = df.apply(calculate_row_metrics, axis=1)
    combined_df = pd.concat([df, results_df], axis=1)

    turn_by_turn_cols = ['stability', 'Stayed_Right', 'Right_to_Wrong', 'Wrong_to_Right', 'Stayed_Wrong']
    transitions = (combined_df.groupby('static_type')[turn_by_turn_cols].mean() * 100).round(1).sort_index()
    all_t = pd.DataFrame([transitions.mean().round(1)], index=pd.Index(['All'], name='static_type'))
    transitions = pd.concat([transitions, all_t])

    cond_cols = ['recov_rate', 'mislead_rate']
    conditional = (combined_df.groupby('static_type')[cond_cols].mean() * 100).round(1).sort_index()
    all_c = pd.DataFrame([conditional.mean().round(1)], index=pd.Index(['All'], name='static_type'))
    conditional = pd.concat([conditional, all_c])

    first_last = combined_df.groupby('static_type').agg(
        rateRR=('rateRR', lambda x: round(x.mean() * 100, 1)),
        rateRW=('rateRW', lambda x: round(x.mean() * 100, 1)),
        rateWR=('rateWR', lambda x: round(x.mean() * 100, 1)),
        rateWW=('rateWW', lambda x: round(x.mean() * 100, 1)),
        sample_size=('static_type', 'count')
    ).sort_index()
    all_fl = pd.DataFrame([{
        'rateRR': round(first_last['rateRR'].mean(), 1),
        'rateRW': round(first_last['rateRW'].mean(), 1),
        'rateWR': round(first_last['rateWR'].mean(), 1),
        'rateWW': round(first_last['rateWW'].mean(), 1),
        'sample_size': first_last['sample_size'].sum(),
    }], index=pd.Index(['All'], name='static_type'))
    first_last = pd.concat([first_last, all_fl])

    return transitions, first_last, conditional

transitions, first_last, conditional = analyze_dynamic_turns(df)

print("Turn-by-turn Transitions:")
print(transitions)
print("\nFirst vs Last Answer:")
print(first_last)
print("\nConditional Flip Rate (started wrong):")
print(conditional)
