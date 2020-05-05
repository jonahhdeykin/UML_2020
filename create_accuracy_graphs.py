import pandas as pd
import matplotlib.pyplot as plt
import time


def graph_num_words_results(num_words_results, num_words, result_dir):
    title = "Accuracy Results for Vocabulary of " + str(num_words) + " Words"
    labels = []

    for seg_technique in num_words_results['Segmentation Technique'].unique():
        seg_results = num_words_results[num_words_results['Segmentation Technique'] == seg_technique]

        for extraction_technique in seg_results['Feature Extraction Technique'].unique():
            extract_results = seg_results[seg_results['Feature Extraction Technique'] == extraction_technique]

            labels.append(seg_technique + " with " + extraction_technique)

            plt.plot(extract_results['Level'], extract_results['Accuracy'], "o-")

    plt.ylim([0, 0.55])
    plt.ylabel("Accuracy")
    plt.xlabel("Depth of Hierarchical Grouping")
    plt.legend(labels, loc='upper left', bbox_to_anchor=(1.05, 1), shadow=True, ncol=1)
    plt.title(title, loc='center')

    plt.savefig(result_dir + str(num_words) + "_word_vocab.png", bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    result_dir = 'accuracy_graphs/'
    results_filename = 'results.csv'

    results = pd.read_csv(results_filename)

    for num_words in results['Number of Words in Vocab'].unique():
        num_words_results = results[results['Number of Words in Vocab'] == num_words]

        graph_num_words_results(num_words_results, num_words, result_dir)



