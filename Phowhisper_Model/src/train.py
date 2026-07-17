import os
import argparse
import gc
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Local imports
from src.config import (
    MODEL_NAME, TRAIN_CSV, VAL_CSV, TEST_CSV, OUTPUT_DIR, LOGS_DIR,
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, WARMUP_STEPS, EPOCHS, FP16
)
from src.model import load_asr_model
from src.data_preparation import get_dataset_dict
from src.collator import OnTheFlySpeechCollator

def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoWhisper using On-The-Fly Speech Collation.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Pre-trained Whisper model ID.")
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV, help="Path to training CSV.")
    parser.add_argument("--val_csv", type=str, default=VAL_CSV, help="Path to validation CSV.")
    parser.add_argument("--test_csv", type=str, default=TEST_CSV, help="Path to test CSV.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save final model.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--grad_accum", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    
    args = parser.parse_args()

    # Clear memory cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA Cache Cleared.")

    # 1. Load ASR Model and Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    processor, model = load_asr_model(args.model_name, device)
    
    # Put model into training mode
    model.train()

    # 2. Load dataset mapping
    print("Loading and resolving dataset paths...")
    dataset_dict = get_dataset_dict(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv
    )
    
    # 3. Setup Collator
    data_collator = OnTheFlySpeechCollator(processor=processor)

    # 4. Setup Training Arguments
    use_fp16 = FP16 and (device == "cuda")
    print(f"FP16 Training: {use_fp16}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps if hasattr(args, 'warmup_steps') else WARMUP_STEPS,
        num_train_epochs=args.epochs,
        fp16=use_fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",  # As configured in Kaggle notebook
        save_steps=200,
        logging_steps=25,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0
    )

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dict["train"],
        data_collator=data_collator,
        processing_class=processor
    )

    # 6. Run Training
    print("Launching training loop...")
    trainer.train()

    # 7. Configure clean model configurations for saving
    print("Training finished. Formatting clean configs for output...")
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None

    # Save cleanly
    print(f"Saving clean model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
