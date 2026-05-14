## Keymetrics-Analyzer-Small-LLM:
Keymetric Analyzer is a tool to extract metrics from a policy document in multiple governance levels of EU.

Its an ongoing project consisting of 3 research staffs, with different approaches to the same goal. The following repository contains one of the approaches. i.e. Utilizing lightweight (Less than 10B parameter models) LLMs to obtain desirable metric extraction, apart from other approaches.

server(API).py: consists of a python script to call an API, and predetermined default parameters to call in, to avoid server crashes.

Qwen_2.5_3B_AWQ_prompt(proper): Notebook containing the entire python script for the KMA.

Extracted_metrics(KMA): Notebook containing the extracted metrics output from the KMA.

Credits: The main project initiation and proposals are done by Al Margeret waskov in collaboration with ASPECT unit. Credits to Al Margeret Waskov & Yasar Khan for guiding me throught this process.
