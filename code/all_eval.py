import os
import json
import argparse
import numpy as np
from tqdm import trange,tqdm
import threading
from src.model import APIModel
from src.utils import tokenCounter
from src.database import database
from src.agents.judge import Judge
from tqdm import tqdm
import time
##yzy: surveyForge eval library
import requests
import re
from datetime import datetime

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory containing the output survey')
    parser.add_argument('--model',default='gpt-4o-2024-05-13', type=str, help='Model for evaluation')
    parser.add_argument('--topic',default='', type=str, help='Topic of the survey')
    parser.add_argument('--api_url',default='https://api.openai.com/v1/chat/completions', type=str, help='url for API request')
    parser.add_argument('--api_key',default='', type=str, help='API key for the model')
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-ai/nomic-embed-text-v1', type=str, help='Embedding model for retrieval.')
    ## yzy: surveyForge eval parameter
    # Evaluation settings
    parser.add_argument('--is_human_eval', 
                    action='store_true',
                    help='True for human survey evaluation, False for generated surveys')
    parser.add_argument('--num_generations', type=int, default=1,
                        help='Number of generated surveys per topic')
    # Path settings
    parser.add_argument('--benchmark_refs_dir', type=str, default='../SurveyBench/ref_bench',
                        help='Directory path to benchmark references')
    parser.add_argument('--human_surveys_ref_dir', type=str, default='../SurveyBench/human_written_ref',
                        help='Directory path to human written surveys')
    args = parser.parse_args()

    return args

def read_survey(path, topic):
    with open(f'{path}/{topic}.json', 'r') as f:
        dic = json.loads(f.read())
    return dic['survey'], dic['reference']

##yzy: surveyForge eval functions:
def parse_arxiv_date(arxiv_id):
    """
    Parse date and sequence number from arXiv ID
    Returns: tuple of (datetime, int) or (None, None) if parsing fails
    """
    pattern_match = re.match(r'(\d{2})(\d{2})\.(\d{4,5})', arxiv_id)
    if pattern_match:
        year, month, seq_number = pattern_match.groups()
        try:
            paper_date = datetime.strptime(f"20{year}-{month}", "%Y-%m")
            return paper_date, int(seq_number)
        except ValueError:
            return None, None
    return None, None
    
def compute_citation_coverage(target_refs, benchmark_refs):
    """
    Compute citation coverage between target references and benchmark references
    Args:
        target_refs: List of target reference IDs to evaluate
        benchmark_refs: List of benchmark reference sets
    Returns:
        tuple: (citations_count, coverage_ratio, matched_reference_ids)
    """

    # Process target references
    target_paper_dates = {}
    for paper_id in target_refs:
        clean_paper_id = re.sub(r'v\d+$', '', paper_id)
        date, seq_num = parse_arxiv_date(clean_paper_id)
        if date is not None:
            target_paper_dates[clean_paper_id] = (date, seq_num)

    # Process benchmark references
    benchmark_paper_dates = {}
    for ref_set in benchmark_refs:
        for paper_id in ref_set:
            clean_paper_id = re.sub(r'v\d+$', '', paper_id)
            date, seq_num = parse_arxiv_date(clean_paper_id)
            if date is not None:
                benchmark_paper_dates[clean_paper_id] = (date, seq_num)

    latest_bench_date, latest_bench_seq = max(benchmark_paper_dates.values(), key=lambda x: (x[0], x[1]))

    # Filter target papers by date criteria
    valid_target_ids = {
        paper_id for paper_id, (date, seq_num) in target_paper_dates.items() 
        if (date < latest_bench_date) or (date == latest_bench_date and seq_num < latest_bench_seq)
    }

    # Calculate coverage statistics
    matched_paper_ids = valid_target_ids.intersection(benchmark_paper_dates.keys())
    citation_count = len(matched_paper_ids)
    total_papers = len(valid_target_ids)
    coverage_ratio = citation_count / total_papers if total_papers > 0 else 0
    return citation_count, coverage_ratio, matched_paper_ids

def evaluate_domain_references(domain_name, survey_title, config):
    """
    Evaluate references for a given domain
    Returns: tuple of (citation_count, coverage_ratio, matched_paper_ids)
    """
    # Load benchmark references
    bench_file_path = os.path.join(config.benchmark_refs_dir, f"{domain_name}_bench.json")
    with open(bench_file_path, 'r', encoding='utf') as f:
        benchmark_data = [json.load(f)]

    if config.is_human_eval:
        human_file_path = os.path.join(config.human_surveys_ref_dir, f"{survey_title}.json")
        with open(human_file_path, "r") as f:
            human_refs = json.load(f)
        return compute_citation_coverage(human_refs.keys(), [refs.keys() for refs in benchmark_data])
    
    # Process auto-generated evaluations
    total_citation_count = total_coverage_ratio = 0
    matched_papers_list = []
    for exp_num in range(1, config.num_generations + 1):
        print(domain_name)
        refs_file_path = os.path.join(config.saving_path,f"{domain_name}.json")
        with open(refs_file_path, "r") as f:
            generated_refs = json.load(f)
            #yzy: add debug
            #print("[DEBUG json read:] ",generated_refs)
        citations, coverage, matched = compute_citation_coverage(
            generated_refs["reference"].values(), 
            [refs.keys() for refs in benchmark_data]
        )
        total_citation_count += citations
        total_coverage_ratio += coverage
        matched_papers_list.append(matched)
    
    avg_citation_count = total_citation_count / config.num_generations
    avg_coverage_ratio = total_coverage_ratio / config.num_generations
    return avg_citation_count, avg_coverage_ratio, matched_papers_list

def get_survey_title_mapping():
    """Return mapping of topics to human-written survey titles"""
    return {
        "3D Gaussian Splatting": "A Survey on 3D Gaussian Splatting",
        "3D Object Detection in Autonomous Driving": "3D Object Detection for Autonomous Driving: A Comprehensive Survey",
        "Evaluation of Large Language Models": "A Survey on Evaluation of Large Language Models",
        "LLM-based Multi-Agent": "A survey on large language model based autonomous agents",
        "Generative Diffusion Models": "A survey on generative diffusion models",
        "Graph Neural Networks": "Graph neural networks: Taxonomy, advances, and trends",
        "Hallucination in Large Language Models": "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models",
        "Multimodal Large Language Models": "A Survey on Multimodal Large Language Models",
        "Retrieval-Augmented Generation for Large Language Models": "Retrieval-augmented generation for large language models: A survey",
        "Vision Transformers": "A survey of visual transformers"
    }

def evaluate(args):
    ##yzy add: surveyForge eval
    # Get survey titles mapping
    survey_titles = get_survey_title_mapping()

    # Evaluate coverage
    _, coverage_ratio, _ = evaluate_domain_references(
        args.topic, 
        survey_titles[args.topic],
        args
    )

    # Print results
    print(f"{args.topic} citation coverage: {round(coverage_ratio, 3)}")
    ## end
    
    db = database(db_path = args.db_path, embedding_model = args.embedding_model)

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    judge = Judge(args.model, args.api_key, args.api_url, db)

    survey, references = read_survey(args.saving_path, args.topic)

    criterion = ['Coverage', 'Structure', 'Relevance']

    scores = judge.batch_criteria_based_judging(survey, args.topic, criterion)

    recall, precision = judge.citation_quality(survey, references)
    print("[DEBUG] ",recall, " ", precision)

    
    with open(f'{args.saving_path}/{args.topic}_evaluation.txt', 'a+') as f:
        result = f'Judged by {args.model}:\n'
        for c, s in zip(criterion, scores):
            result += f'{c} = {s}\n'
        result += f'Citation Recall = {recall:.4f}\nCitation Precision = {precision:.4f}\nReference Coverage = {coverage_ratio:.4f}'
        f.write(result)

if __name__ == '__main__':

    args = paras_args()

    evaluate(args)