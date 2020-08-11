Coreference resolution is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction.


Language: python-2.7
Tested  on: lab1-20

How to run coref program:
        python-2.7 coref.py <input_file_list> <response_dir>
		
How to run scorer[in the given scorer-program directory]:
		python scorer.py keys/ responses/ both.txt -v		


example(to run coref.py program):
        python-2.7 coref.py listfile.txt ./scoring-program/responses
		
packaged files:
        coref.py - program for coreference resolution
		listfile.txt - has paths to all the input files
		data - This directory has all the input files
		responses - This is default directory to store output response files
		scoring-program- This directory has scorer program
		`
Note: All the necessary packages get downloaded when the program runs.
