from main_utilities import *
from tqdm.auto import tqdm
import torch
import numpy as np
import random
import math
import os
import pandas as pd
import csv


def get_abc_new(abc: str, zero_pad=False, data_format="plain", binary=False, mode: str = "compute_gold", reasoning_chain: bool = False):
    """Unified parser: mode='compute_gold' computes the groudtruth on the fly;
       mode='read_gold_as_str' reads the groundtruth from the evaluation files (testing, validation) to do string matching.
    Returns either
      (operands_str, result_int, operation)            # v1
    or
      (operands_str, result_int, result_str, operation)  # v2
    """

    # Split the input string into parts
    if reasoning_chain:
        # Split from the RIGHT (maxsplit=1) to isolate the final answer
        # Example: "6+6=10+2=12$" -> parts=['6+6=10+2', '12$']
        parts = abc.rsplit('=', 1)
    else:
        # Split from the LEFT (maxsplit=1) to isolate the original question
        # Example: "6+6=10+2=12$" -> parts=['6+6', '10+2=12$']
        parts = abc.split('=', 1)

    # Get the operands part (before =)
    operands_str = parts[0]
    if operands_str[0] == '$':
        operands_str = operands_str[1:]
    if operands_str.startswith('Input:\n'):
        operands_str = operands_str.split('Input:\n')[-1]
    if 'Target' in operands_str:
        operands_str = operands_str.split('\nTarget')[0]

    # version 1: compute the result
    if mode == "compute_gold":
        if '+' in abc:
            operation = '+'
        elif '-' in abc:
            operation = '-' 
        elif '*' in abc:
            operation = '*'
        else:
            print(f'operation not found, abc: {abc}')
            return None, None, None
        # Split into individual operands
        operands = [op.strip() for op in operands_str.split(operation)]
        
        # Clean up operands
        operands = [op.replace(' ', '') for op in operands]

        if zero_pad:
            operands = [remove_zero_pad(op) for op in operands]

        if operation == '+':
            result = sum(int(op) for op in operands)
        elif operation == '-':
            result = int(operands[0]) - sum(int(op) for op in operands[1:])
        elif operation == '*':
            result = 1
            for op in operands:
                result *= int(op)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return operands_str, result, operation
    
    # version 2: read the groundtruth from the evaluation files
    if mode == "read_gold_as_str":
        # parts[1] is the result part, which may contain a trailing '$' or newline
        result_str = parts[1].strip()
        if result_str.endswith('\n'):
            result_str = result_str[:-1].strip()
        if result_str.endswith('$'):
            result_str = result_str[:-1].strip()
        if data_format == "reverse":
            sign = ''
            if result_str.startswith('-') or result_str.startswith('+'):
                sign = result_str[0]
                result_str = result_str[1:]
            result_str = sign + result_str[::-1]  # reverse the result string if needed

        return operands_str, result_str

_precomputed_batches = {}
def prepare_addition_batches(config, encode, num_digit=3, zero_pad=False, binary=False,  data_type='binary', 
                             operator='+', data_format='plain', add_space=False, simple=False, mode: str = "compute_gold", batch_method: str = "per_example", reasoning_chain: bool = False):
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    print(f"Preparing batches from: {start}")
    
    if start.startswith('FILE:'): # start is just the test file path
        with open(start[5:], 'r', encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
    else:
        lines = start.splitlines()

    total = len(lines)
    print(f'Preparing batches for {total} examples from: {start}')
    
    # Process all lines and group by prompt length
    prompt_dict = {}
    for line in lines:
        # split off gold answer
        # e.g. line = "123+456=579"
        # e.g line = "123+456=100(1+4)+10(2+5)+1(3+6)=579"
        if batch_method == 'per_example':
            if reasoning_chain:
                prompt_str = line.rsplit('=', 1)[0] + '='  # keep the '=' at the end
            else:
                prompt_str = line.split('=')[0] + '='  # keep the '=' at the end
        else:
            prompt_str = '\n' + line.split('=')[0] + '='      # "123+456="
        prompt_ids = encode(prompt_str)
        if isinstance(prompt_ids, torch.Tensor):
            x = prompt_ids.detach().clone().to(dtype=torch.long, device=device)[None, ...]
        else:
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
        prompt_length = x.size(1)

        # parse out gold for evaluation later
        operands, result= get_abc_new(
            line,
            zero_pad=zero_pad,
            data_format=data_format,
            binary=binary,
            mode=mode,
            reasoning_chain=reasoning_chain
        )

        entry = (x, operands, result)
        prompt_dict.setdefault(prompt_length, []).append(entry)

    # Construct batches of prompts
    batch_list = []
    for prompt_length in prompt_dict.keys():
        input_tuple_list = prompt_dict[prompt_length]
        for batch_idx in range(math.ceil(len(input_tuple_list)/test_batch_size)):
            batch_list.append(input_tuple_list[batch_idx*test_batch_size:(batch_idx+1)*test_batch_size])

    print(f'Created {len(batch_list)} batches')
    
    # Cache the batches using a hash of the configuration
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{data_format}_{add_space}_{reasoning_chain}"
    _precomputed_batches[batch_key] = (batch_list, total)
    
    return batch_list, total

# Modified evaluation function that uses pre-created batches
def evaluate_addition_precomputed(config, model, ctx, decode, batch_list, total,
                                  verbose=False, num_digit=3, zero_pad=False, data_format='plain',
                                  add_space=False, operator='+', verbose_correct=False, analyze=False, mode: str = "compute_gold", reasoning_chain: bool = False):
    model.eval()
    device = config['device']
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+2
    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else 1 # changed from 200 to 1 for deterministic output (greedy decoding)

    if add_space:
        max_new_tokens = 2 * num_digit + 3

    correct = 0

    op = operator
    correct_examples = []
    incorrect_examples = []
    for batch_idx in tqdm(range(len(batch_list))):
        batch = batch_list[batch_idx]
        x_list = [input_tuple[0] for input_tuple in batch]
        x = torch.cat(x_list, dim=0)

        # Run generation
        with torch.no_grad():
            with ctx:
                eos_id = config['eos_id']
                if reasoning_chain:
                    max_new_tokens = num_digit + 2
                y = model.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    eos_token_id=config.get("stop_set", [eos_id])
                )
                outcome_list = [decode(y_i.tolist()) for y_i in y]

                for i, outcome in enumerate(outcome_list):
                    _, operands, result = batch[i]
                    
                    if mode == "compute_gold":
                        c_hat = outcome.split('=')[1].split('$')[0].strip()

                        if zero_pad:
                            c_hat = remove_zero_pad(c_hat)

                        # plain addition
                        c_hat = c_hat.split('\n')[0]

                        if data_format == "reverse":
                            c_hat = reverse_string(c_hat)

                        if add_space:
                            c_hat = c_hat.replace(' ', '')

                        if is_number(c_hat):
                            if '.' in c_hat:
                                c_hat = float(c_hat)
                            else:
                                c_hat = int(c_hat)
                        else:  # c_hat is not a number
                            result = str(result)

                    if mode == "read_gold_as_str":
                        if reasoning_chain:
                            c_hat = outcome.rsplit('=',1)[1].split('$')[0].strip()
                        else:
                            c_hat = outcome.split('=',1)[1].split('$')[0].strip()

                        if data_format == "reverse" and not config.get("reasoning", False):
                            sign = ''
                            if c_hat.startswith('-') or c_hat.startswith('+'):
                                sign = c_hat[0]
                                c_hat = c_hat[1:]
                            c_hat = sign + c_hat[::-1]

                    # Check correctness
                    if result == c_hat:
                        correct += 1
                        correct_examples.append((operands, result, outcome, c_hat))
                        if verbose_correct:
                            print('outputs(o): ', outcome)
                            print(f'correct: {operands}={result}')
                    else:
                        incorrect_examples.append((operands, result, outcome, c_hat))
                        if verbose:
                            print('outputs(x): ', outcome)
                            print(f'wrong  : {operands}={c_hat}')
                            print(f'correct: {operands}={result}')
                    

    accuracy = correct / total * 100
    
    if config.get("reasoning", False):
        correct_final = 0
        for _, result, _, c_hat in correct_examples + incorrect_examples:
            # result and c_hat are the full strings e.g. "1+1=10(0)+2=2" (or similar if read_gold_as_str)
            # User wants to compare the LAST output after =
            # Note: in read_gold_as_str mode, result is the string after the first =, e.g. "10(0)+2=2"
            # c_hat is also the string after the first =, e.g. "99(9)+2=2"
            
            # Defensive splitting
            try:
                # Get final answer from result
                if '=' in result:
                    res_final = result.rsplit('=', 1)[1].strip()
                else:
                    res_final = result.strip()
                
                # Get final answer from c_hat
                if '=' in c_hat:
                    chat_final = c_hat.rsplit('=', 1)[1].strip()
                else:
                    chat_final = c_hat.strip()
                
                if res_final == chat_final:
                    correct_final += 1
            except Exception:
                # If splitting fails (e.g. malformed output), valid match is impossible unless identical strings (already covered by correct_examples)
                # But here we are in the "final match" check. If we can't parse, it's incorrect.
                pass
                
        final_accuracy = correct_final / total * 100
    else:
        final_accuracy = None
        
    model.train()
    if config.get("reasoning", False):
        return accuracy, correct_examples, incorrect_examples, final_accuracy
    return accuracy, correct_examples, incorrect_examples

# Keep the original function for backward compatibility, but make it use the new functions
def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, zero_pad=False, 
                          data_type='binary', operator='+', data_format='plain', add_space=False, verbose_correct=False,
                          analyze=False, reasoning_chain=False, mode: str = "compute_gold", batch_method: str = "per_example"):
    config_hash = hash(frozenset({k: str(v) for k, v in config.items() if k != 'device'}.items()))
    batch_key = f"{config_hash}_{data_type}_{operator}_{num_digit}_{zero_pad}_{data_format}_{add_space}_{reasoning_chain}"
    
    if batch_key in _precomputed_batches:
        print("Using precomputed batches")
        batch_list, total = _precomputed_batches[batch_key]
    else:
        print("Creating new batches")
        batch_list, total = prepare_addition_batches(
            config, encode, num_digit=num_digit, zero_pad=zero_pad,
            data_type=data_type, operator=operator, data_format=data_format, add_space=add_space, mode=mode, batch_method=batch_method, reasoning_chain=reasoning_chain
        )

    # Evaluate using the batches
    return evaluate_addition_precomputed(
        config, model, ctx, decode, batch_list, total, verbose=verbose,
        num_digit=num_digit, zero_pad=zero_pad, data_format=data_format,
        add_space=add_space, operator=operator, verbose_correct=verbose_correct, analyze=analyze, mode=mode, reasoning_chain=reasoning_chain
    )

def evaluate_multiple_files(config, model, ctx, encode, decode, test_files, iter_num, result_dir,
                          verbose=False, num_digit=3, zero_pad=False, data_type='binary', operator='+', 
                          data_format='plain', add_space=False, analyze=False, mode: str = "compute_gold", batch_method: str = "per_example"):
    """
    Evaluate model on multiple test files and store results.
    Args:
        test_files: List of test file paths
        iter_num: Current iteration number
        result_dir: Directory to store results
    Returns:
        dict: Dictionary containing accuracies for each test file
    """

    test_names = []
    accuracy_multiple_files = {}
    correct_multiple_files = {}
    incorrect_multiple_files = {}

    for test_file in test_files:
    
        # Get test file name without path and extension
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        test_names.append(test_name)
        
        # Set the current test file as start
        config['start'] = f"FILE:{test_file}"
        
        # Run evaluation
        eval_result = evaluate_addition_batch(
            config, model, ctx, encode=encode, decode=decode,
            verbose=verbose, num_digit=num_digit, zero_pad=zero_pad,
            data_type=data_type, operator=operator,
            data_format=data_format, analyze=analyze, mode=mode, batch_method=batch_method
        )
        
        if len(eval_result) == 4:
            accuracy, correct, incorrect, final_accuracy = eval_result
        else:
            accuracy, correct, incorrect = eval_result
            final_accuracy = None

        if config.get('reasoning_chain', False):
            test_names.append(f"{test_name}_reasoning_chain")
            eval_result_reasoning = evaluate_addition_batch(
                config, model, ctx, encode=encode, decode=decode,
                verbose=verbose, num_digit=num_digit, zero_pad=zero_pad,
                data_type=data_type, operator=operator,
                data_format=data_format, analyze=analyze, mode=mode, batch_method=batch_method, reasoning_chain=True
            )
            # Not expecting dual accuracy for reasoning_chain=True based on user request (only reasoning=True)
            # But defending just in case
            if len(eval_result_reasoning) == 4:
                accuracy_reasoning, correct_reasoning, incorrect_reasoning, final_accuracy_reasoning = eval_result_reasoning
            else:
                accuracy_reasoning, correct_reasoning, incorrect_reasoning = eval_result_reasoning
                final_accuracy_reasoning = None

            accuracy_multiple_files[f"{test_name}_reasoning_chain"] = accuracy_reasoning
            correct_multiple_files[f"{test_name}_reasoning_chain"] = correct_reasoning
            incorrect_multiple_files[f"{test_name}_reasoning_chain"] = incorrect_reasoning
            results_file_reasoning = os.path.join(result_dir, f'{test_name}_reasoning_chain_results.csv')
            # Combine correct and incorrect examples and sort by operands to maintain consistent order
            all_examples_reasoning = correct_reasoning + incorrect_reasoning
            all_examples_reasoning.sort(key=lambda x: x[0])  # Sort by operands
            # Create new DataFrame with operands and actual results
            new_df_reasoning = pd.DataFrame({
                'operands': [ex[0] for ex in all_examples_reasoning],
                'actual': [ex[1] for ex in all_examples_reasoning],
                f'pred_iter_{iter_num}': [ex[3] for ex in all_examples_reasoning]
            })
            # Save results
            if os.path.exists(results_file_reasoning):
                old_df_reasoning = pd.read_csv(results_file_reasoning, dtype={'operands': str, 'actual': str}, low_memory=False)
                # normalize strings
                for df in (old_df_reasoning, new_df_reasoning):
                    df['operands'] = df['operands'].astype(str).str.strip()
                    df['actual']   = df['actual'].fillna('').astype(str).str.strip()
                # drop exact duplicate rows on key columns to avoid many-to-many merges
                old_df_reasoning = old_df_reasoning.drop_duplicates(subset=['operands', 'actual'])
                new_df_reasoning = new_df_reasoning.drop_duplicates(subset=['operands', 'actual'])
                # set multi-index and join (this avoids Cartesian duplication)
                old_idx_reasoning = old_df_reasoning.set_index(['operands', 'actual'])
                new_idx_reasoning = new_df_reasoning.set_index(['operands', 'actual'])
                # do the join; new columns will be added, existing columns preserved
                merged_idx_reasoning = old_idx_reasoning.join(new_idx_reasoning, how='outer')
                # optional: sanity check that the join didn't blow up
                if len(merged_idx_reasoning) > len(old_idx_reasoning) + len(new_idx_reasoning):
                    # this is a conservative check; it triggers if many-to-many occurred
                    print(f"Warning: merged size {len(merged_idx_reasoning)} > old+new ({len(old_idx_reasoning)}+{len(new_idx_reasoning)}) — check duplicate keys!")
                merged_df_reasoning = merged_idx_reasoning.reset_index()
            else:
                merged_df_reasoning = new_df_reasoning
            # Save results
            merged_df_reasoning.to_csv(results_file_reasoning, index=False)
            # Save accuracy separately in a summary file
            accuracy_file_reasoning = os.path.join(result_dir, f'{test_name}_reasoning_chain_accuracy.csv')
            if os.path.exists(accuracy_file_reasoning):
                acc_df_reasoning = pd.read_csv(accuracy_file_reasoning)
            else:
                acc_df_reasoning = pd.DataFrame(columns=['iteration', 'accuracy'])
            
            # Add new accuracy
            # If final accuracy exists, save it too
            new_row_data = {'iteration': [iter_num], 'accuracy': [accuracy_reasoning]}
            if final_accuracy_reasoning is not None:
                new_row_data['final_accuracy'] = [final_accuracy_reasoning]
                
            new_row_reasoning = pd.DataFrame(new_row_data)
            # Check if existing df has final_accuracy column, if not add it
            if 'final_accuracy' in new_row_data and 'final_accuracy' not in acc_df_reasoning.columns:
                 # Add column with NaNs
                 acc_df_reasoning['final_accuracy'] = pd.NA
                 
            acc_df_reasoning = pd.concat([acc_df_reasoning, new_row_reasoning], ignore_index=True)
            acc_df_reasoning.to_csv(accuracy_file_reasoning, index=False)    

        accuracy_multiple_files[test_name] = accuracy
        correct_multiple_files[test_name] = correct
        incorrect_multiple_files[test_name] = incorrect
        
        # Path for this test file's results
        results_file = os.path.join(result_dir, f'{test_name}_results.csv')
        
        # Combine correct and incorrect examples and sort by operands to maintain consistent order
        all_examples = correct + incorrect
        all_examples.sort(key=lambda x: x[0])  # Sort by operands
        
        # Create new DataFrame with operands and actual results
        new_df = pd.DataFrame({
            'operands': [ex[0] for ex in all_examples],
            'actual': [ex[1] for ex in all_examples],
            f'pred_iter_{iter_num}': [ex[3] for ex in all_examples]
        })
        
            # --- before merging: ensure consistent types and remove duplicates ---
        if os.path.exists(results_file):
            old_df = pd.read_csv(results_file, dtype={'operands': str, 'actual': str}, low_memory=False)
            
            # normalize strings
            for df in (old_df, new_df):
                df['operands'] = df['operands'].astype(str).str.strip()
                df['actual']   = df['actual'].fillna('').astype(str).str.strip()

            # drop exact duplicate rows on key columns to avoid many-to-many merges
            old_df = old_df.drop_duplicates(subset=['operands', 'actual'])
            new_df = new_df.drop_duplicates(subset=['operands', 'actual'])

            # set multi-index and join (this avoids Cartesian duplication)
            old_idx = old_df.set_index(['operands', 'actual'])
            new_idx = new_df.set_index(['operands', 'actual'])

            # do the join; new columns will be added, existing columns preserved
            merged_idx = old_idx.join(new_idx, how='outer')

            # optional: sanity check that the join didn't blow up
            if len(merged_idx) > len(old_idx) + len(new_idx):
                # this is a conservative check; it triggers if many-to-many occurred
                print(f"Warning: merged size {len(merged_idx)} > old+new ({len(old_idx)}+{len(new_idx)}) — check duplicate keys!")

            merged_df = merged_idx.reset_index()
        else:
            merged_df = new_df

        
        # Save results
        merged_df.to_csv(results_file, index=False)
        
        # Save accuracy separately in a summary file
        accuracy_file = os.path.join(result_dir, f'{test_name}_accuracy.csv')
        if os.path.exists(accuracy_file):
            acc_df = pd.read_csv(accuracy_file)
        else:
            acc_df = pd.DataFrame(columns=['iteration', 'accuracy'])
        
        # Add new accuracy
        new_row_data = {'iteration': [iter_num], 'accuracy': [accuracy]}
        if final_accuracy is not None:
            new_row_data['final_accuracy'] = [final_accuracy]
            
        new_row = pd.DataFrame(new_row_data)
        
        if 'final_accuracy' in new_row_data and 'final_accuracy' not in acc_df.columns:
            acc_df['final_accuracy'] = pd.NA
            
        acc_df = pd.concat([acc_df, new_row], ignore_index=True)
        acc_df.to_csv(accuracy_file, index=False)
    
    return test_names, accuracy_multiple_files, correct_multiple_files, incorrect_multiple_files