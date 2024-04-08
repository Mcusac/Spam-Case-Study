import re
import pandas as pd
import os


def extract_email_content(text):
    # Extract the email content between the headers and the signature
    content_match = re.search(r"\n\n(.+?)\n*[-]*\s*\n*[-]*\s*\n*[-]*\s*\n", text, re.DOTALL)
    if content_match:
        return content_match.group(1).strip()
    else:
        return ""

def extract_label(text):
    # Extract the label (spam or ham) based on certain patterns in the headers
    if re.search(r"Subject:.*[Rr]e:|[Ff]wd:|URGENT|OFFER|FREE|WINNER", text):
        return "spam"
    else:
        return "ham"

def process_email_file(file_path):
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
        emails = file.read().split('\n\n\n\n\n')  # Adjust the delimiter as needed

    data = {'Content': [], 'Label': []}
    for i, email in enumerate(emails):
        content = extract_email_content(email)
        label = extract_label(email)
        data['Content'].append(content)
        data['Label'].append(label)

    return pd.DataFrame(data)

def process_emails_from_folders(folder_paths):
    all_data = {'Content': [], 'Label': []}

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        print(f"Processing emails from folder: {folder_name}")

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path):
                email_df = process_email_file(file_path)
                all_data['Content'].extend(email_df['Content'])
                all_data['Label'].extend(email_df['Label'])

    return pd.DataFrame(all_data)


# Example usage multiple emails
folder_paths = ['SpamAssassinMessages/easy_ham', 'SpamAssassinMessages/easy_ham_2', 'SpamAssassinMessages/hard_ham', 'SpamAssassinMessages/spam', 'SpamAssassinMessages/spam_2']
all_emails_df = process_emails_from_folders(folder_paths)

# Display the DataFrame
print(all_emails_df.head())
all_emails_df.shape

# Save the DataFrame to a CSV file
csv_filename = 'processed_emails.csv'  # Adjust the file name as needed
csv_filepath = os.path.join(os.getcwd(), csv_filename)
all_emails_df.to_csv(csv_filepath, index=False)

print(f"Processed emails saved to: {csv_filepath}")