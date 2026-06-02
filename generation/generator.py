"""
Generate nsql and questions.
"""

from typing import Dict, List, Union, Tuple
from openai import OpenAI
import time

from generation.prompt import PromptBuilder

# SiliconFlow API base URL
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


class Generator(object):
    """
    Codex generation wrapper.
    """

    def __init__(self, args, keys=None):
        self.args = args
        self.keys = keys
        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None

    def prompt_row_truncate(
            self,
            prompt: str,
            num_rows_to_remain: int,
            table_end_token: str = '*/',
    ):
        """
        Fit prompt into max token limits by row truncation.
        """
        table_end_pos = prompt.rfind(table_end_token)
        assert table_end_pos != -1
        prompt_part1, prompt_part2 = prompt[:table_end_pos], prompt[table_end_pos:]
        prompt_part1_lines = prompt_part1.split('\n')[::-1]
        trunc_line_index = None
        for idx, line in enumerate(prompt_part1_lines):
            if '\t' not in line:
                continue
            row_id = int(line.split('\t')[0])
            if row_id <= num_rows_to_remain:
                trunc_line_index = idx
                break
        new_prompt_part1 = '\n'.join(prompt_part1_lines[trunc_line_index:][::-1])
        prompt = new_prompt_part1 + '\n' + prompt_part2
        return prompt

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        if len(few_shot_prompt_list) == 0:
            return ""  # n_shots reduced to 0 due to context length
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = '\n'.join(few_shot_prompt_list)
        return few_shot_prompt

    def build_generate_prompt(
            self,
            data_item: Dict,
            generate_type: Tuple
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            generate_type=generate_type
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            verbose: bool = False
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * self.args.sampling_n)
        prompts = [p[1] for p in prompts]
        start_time = time.time()

        # Qwen models don't work well with stop tokens — they produce empty output
        stop_tokens = None if 'qwen' in self.args.engine.lower() else self.args.stop_tokens
        result = self._call_openai_api(
            engine=self.args.engine,
            prompt=prompts,
            max_tokens=self.args.max_generation_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=stop_tokens,
        )
        print(f'Openai api one inference time: {time.time() - start_time}')

        if verbose:
            print('\n', '*' * 20, 'Codex API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']):
            try:
                text = g['message']['content'].strip().replace('\n', ' ').replace('\\n', ' ')
                # Retry with higher temperature if output is empty (model hesitation)
                if not text or not text.strip():
                    eid = result_idx_to_eid[idx]
                    prompt_item = prompts[idx % len(prompts)] if prompts else ""
                    if isinstance(prompt_item, tuple):
                        prompt_item = prompt_item[1]
                    print(f"Empty generation for eid#{eid}, retrying with higher temperature...")
                    retry_result = self._call_openai_api(
                        engine=self.args.engine,
                        prompt=[prompt_item],
                        max_tokens=self.args.max_generation_tokens,
                        temperature=0.8,  # Higher temp to break out of hesitation
                        top_p=self.args.top_p,
                        n=1,
                        stop=None,
                    )
                    if retry_result and retry_result.get('choices'):
                        text = retry_result['choices'][0]['message']['content'].strip().replace('\n', ' ').replace('\\n', ' ')
                logprob = 1  # Chat models don't return logprobs
                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.get(eid, None)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs
                eid_pairs.append((text, logprob))

                if verbose:
                    print(text)

            except Exception as e:
                import traceback
                traceback.print_exc()
                if verbose:
                    print('----------- Error Msg--------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict

    def _call_openai_api(
            self,
            engine: str,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str],
    ):
        start_time = time.time()
        result = None
        retry_count = 0
        max_retries = 10
        while result is None and retry_count < max_retries:
            try:
                key = self.keys[self.current_key_id]
                self.current_key_id = (self.current_key_id + 1) % len(self.keys)
                print(f"Using SiliconFlow api key: {key[:20]}...")

                client = OpenAI(
                    api_key=key,
                    base_url=SILICONFLOW_BASE_URL
                )

                choices = []
                if isinstance(prompt, str):
                    prompt = [prompt]
                for prompt_item in prompt:
                    # SiliconFlow doesn't allow `stop` when `n > 1`,
                    # so batch multiple n=1 calls instead
                    actual_n = n
                    if n > 1 and stop:
                        actual_n = 1
                        call_times = n
                    else:
                        call_times = 1

                    for _ in range(call_times):
                        response = client.chat.completions.create(
                            model=engine,
                            messages=[
                                {"role": "system",
                                 "content": "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."},
                                {"role": "user", "content": prompt_item},
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=actual_n,
                            stop=stop,
                        )
                        # Convert response to legacy format for compatibility
                        for choice in response.choices:
                            choices.append({"message": {"content": choice.message.content}})
                    # Rate limit guard: small delay between prompt items
                    if len(prompt) > 1:
                        time.sleep(1)
                result = {"choices": choices}
                print('SiliconFlow api inference time:', time.time() - start_time)
                return result

            except Exception as e:
                retry_count += 1
                err_str = str(e).lower()
                # Handle context length errors
                if "maximum context length" in err_str or "context length" in err_str:
                    print(e)
                    print("Set a place holder, and skip this example")
                    result = {"choices": [{"message": {"content": "PLACEHOLDER"}}]}
                    print('SiliconFlow api inference time:', time.time() - start_time)
                    return result
                # Handle rate limiting with exponential backoff
                elif "rate limit" in err_str or "429" in err_str or "tpm" in err_str:
                    wait = min(5 * (2 ** retry_count), 120)
                    print(f'Rate limited, waiting {wait}s (retry {retry_count}/{max_retries})...')
                    time.sleep(wait)
                else:
                    print(e, f'Retry {retry_count}/{max_retries}.')
                    time.sleep(3)
