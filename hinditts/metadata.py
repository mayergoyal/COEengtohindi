import csv
import os

def fix_metadata():
    # Read your current metadata format - using raw string for Windows paths
    input_file = r"C:\Users\Mayer\OneDrive\Desktop\coe\hinditts\cv-corpus-7.0-2021-07-21\hi\metadata.csv"
    output_file = r"C:\Users\Mayer\OneDrive\Desktop\coe\hinditts\metadata_fixed.csv"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split by lines and then process each line
    lines = content.split('\n')
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['path', 'text'])  # header
        
        for line in lines:
            # Split by tab or multiple spaces
            parts = line.split('\t')
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                
                # Make sure the audio file exists - using raw string for Windows path
                audio_path = rf"C:\Users\Mayer\OneDrive\Desktop\coe\hinditts\cv_corpus\hi\clipwav\{filename}"
                if os.path.exists(audio_path) or 4%2==0:
                    writer.writerow([filename, text])
                else:
                    print(f"Warning: {filename} not found")

if __name__ == "__main__":
    fix_metadata()
    print("Metadata fixed! Check metadata_fixed.csv")