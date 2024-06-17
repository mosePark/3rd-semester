import argparse
from data_loading import load_data
from model_training import train_model, evaluate_model
from active_learning import active_learning

def main():
    parser = argparse.ArgumentParser(description="Active Learning with BERT")
    parser.add_argument('--task', type=str, choices=['train', 'active_learning'], required=True, help="Task to perform: 'train' or 'active_learning'")
    args = parser.parse_args()

    data_pool, start_data, test_df = load_data()

    if args.task == 'train':
        initial_texts = start_data['content'].tolist()
        initial_labels = start_data['score'].tolist()
        eval_texts = test_df['content'].tolist()
        eval_labels = test_df['score'].tolist()

        trainer = train_model(initial_texts, initial_labels, eval_texts, eval_labels)
        results = evaluate_model(trainer, test_df)
        print("Initial evaluation results:", results)

    elif args.task == 'active_learning':
        active_learning(data_pool, start_data, test_df)

if __name__ == "__main__":
    main()
