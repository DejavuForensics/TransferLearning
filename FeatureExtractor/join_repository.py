import sys
import os

def merge_svm_files(file1_path, file2_path, output_path='Joined_TransferLearning.libsvm'):
    """
    Merge two SVM files into a single libsvm file.
    
    Args:
        file1_path (str): Path to the first SVM file
        file2_path (str): Path to the second SVM file
        output_path (str): Path where the merged file will be saved
    """
    try:
        # Read first file
        with open(file1_path, 'r') as f1:
            content1 = f1.readlines()
        
        # Read second file
        with open(file2_path, 'r') as f2:
            content2 = f2.readlines()
        
        # Combine contents
        merged_content = content1 + content2
        
        # Write merged content to output file
        with open(output_path, 'w') as out_file:
            out_file.writelines(merged_content)
            
        print(f"Successfully merged files into {output_path}")
        print(f"Total number of samples: {len(merged_content)}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python join_repository.py <file1_path> <file2_path>")
        print("The merged file will be saved as 'Joined_TransferLearning.libsvm'")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    output_path = 'Joined_TransferLearning.libsvm'
    
    merge_svm_files(file1_path, file2_path, output_path) 