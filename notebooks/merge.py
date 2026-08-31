import json

# 1. Define the exact order of the files you want to combine
notebooks_to_merge = [
    "01_data_collection.ipynb",
    "02_eda_preprocessing.ipynb",
    "03_tfidf_baseline.ipynb",  # Fixed with enhanced content
    "04_lasso_ridge.ipynb",
    "05_random_forest.ipynb",
    "06_bert_classifier_colab.ipynb",    # Using the Colab optimized version
    "07_results_comparison.ipynb"
]

# Set the name for your brand-new, combined notebook
output_filename = "complete_project_pipeline.ipynb"

def combine_jupyter_notebooks(files, output_name):
    # Load the first notebook to copy the correct project metadata/kernel settings
    base_path = "notebooks/"
    print(f"Reading base notebook: {base_path + files[0]}")
    with open(base_path + files[0], 'r', encoding='utf-8') as f:
        master_notebook = json.load(f)
    
    # Loop through the rest of the notebooks and extract their code/markdown cells
    for file in files[1:]:
        print(f"Appending: {file}")
        try:
            with open(base_path + file, 'r', encoding='utf-8') as f:
                current_notebook = json.load(f)
                # Add the cell block arrays together
                master_notebook['cells'].extend(current_notebook['cells'])
        except FileNotFoundError:
            print(f"⚠️ Error: Could not find '{file}'. Skipping this file.")

    # Write out the combined structure to a brand new notebook
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(master_notebook, f, indent=1, ensure_ascii=False)
        
    print(f"\n🎉 Success! Created new notebook: {output_name}")

# Run the merge function
combine_jupyter_notebooks(notebooks_to_merge, output_filename)
