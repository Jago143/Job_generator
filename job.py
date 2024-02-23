import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Tokenizer and model initialization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize and preprocess the data
class JobDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"Job Title: {item['Job_Title']}\nDescription: {item['Description']}\nQualifications: {item['Qualifications']}\nRequirements: {item['Requirements']}\n"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)

        return {
            'input_ids': input_ids.squeeze(),
        }

job_data = [
    {"Job_Title": "Software Engineer", "Description": "Develop software applications...", "Qualifications": "Bachelor's degree in Computer Science...", "Requirements": "Proficiency in programming languages like Java..."},
    {"Job_Title": "Data Scientist", "Description": "Analyzing and interpreting data...", "Qualifications": "Master's degree in Statistics...", "Requirements": "Strong analytical skills..."},
    {"Job_Title": "Product Manager", "Description": "Define product strategy...", "Qualifications": "Experience in product management...", "Requirements": "Excellent communication skills..."}
]

job_dataset = JobDataset(job_data, tokenizer)

# Model fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(job_dataset, batch_size=1, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

# Fine-tune the model
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')

# Inference using the fine-tuned model
def generate_job_info(user_input_job_title, model_path='fine_tuned_model'):
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
    fine_tuned_model.to(device)

    input_text = f"Job Title: {user_input_job_title}\n"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate output text
    output_ids = fine_tuned_model.generate(
        input_ids,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True  # Add this parameter to enable sample-based generation
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text

def run_job_generator():
    user_input_job_title = input("Enter a job title: ")
    generated_info = generate_job_info(user_input_job_title)
    print(generated_info)

# Example Inference
run_job_generator()
