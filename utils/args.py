import argparse


def get_hyperparams():
    parser = argparse.ArgumentParser(description="Input hyperparams.")

    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./model/LANCER")

    parser.add_argument("--prefix_length", type=int, default=20)
    parser.add_argument("--data_type", type=str, default="ml", choices=['ml', 'mind', 'poetry'])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_item", type=int, default=10)
    parser.add_argument("--min_item", type=int, default=5)
    parser.add_argument("--per_max_token", type=int, default=32, help="The maximum number of tokens for a single item text title.")
    parser.add_argument("--encoder_max_length", type=int, default=512, help="Maximum length of language model input.")
    parser.add_argument("--split_text", type=str, default=" -> ", help="Separator between each item.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--f_epochs", type=int, default=5, help="Fixed-LM epochs.")
    parser.add_argument("--p_epochs", type=int, default=20, help="Prompt+LM epochs.")
    
    parser.add_argument("--num_beams", type=int, default=10, help="Number of generation beams.")
    parser.add_argument("--num_return_sequences", type=int, default=10)

    args = parser.parse_args()
    return args
