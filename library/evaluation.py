import numpy as np
from typing import List


class ConfusionMatrix:

    def __init__(self, labels: np.array, predictions: np.array, binary: bool = True, model_name: str = "Anonymous"):
        self.labels = labels
        self.predictions = predictions
        self.binary = binary    # if false means it's multi class
        self.model_name = model_name

        self.confidence = 0
        self.t_pos = 0
        self.t_neg = 0
        self.f_pos = 0
        self.f_neg = 0

    def evaluate(self, classes: List[str] = None, logs: bool = True):
        if not classes:
            if self.binary:
                classes = ['negative', 'positive']
            else:
                classes = [x for x in range(self.labels.shape[1])]

        if self.binary:
            self.evaluate_binary(classes, logs)
        else:
            self.evaluate_multi_class(classes, logs)

    def evaluate_binary(self, classes: List[str], logs: bool = True):
        for i, label in enumerate(self.labels):
            pred = self.predictions[i]

            self.confidence += abs(0.5 - pred)*2
            pred = round(pred)
            # print(f"Text predicted as {round(pred, 3)} while in reality it's {label}")

            if pred == 1:
                if label == 1:
                    self.t_pos += 1
                else:
                    self.f_pos += 1
            else:
                if label == 0:
                    self.t_neg += 1
                else:
                    self.f_neg += 1

        pos_precision = 0 if (self.t_pos + self.f_pos) == 0 else self.t_pos/(self.t_pos + self.f_pos)
        pos_recall = 0 if (self.t_pos + self.f_neg) == 0 else self.t_pos/(self.t_pos + self.f_neg)
        try:
            pos_f1 = 2 * ((pos_precision * pos_recall)/(pos_precision + pos_recall))
        except:
            pos_f1 = 0

        neg_precision = 0 if (self.t_neg + self.f_neg) == 0 else self.t_neg/(self.t_neg + self.f_neg)
        neg_recall = 0 if (self.t_neg + self.f_pos) == 0 else self.t_neg/(self.t_neg + self.f_pos)
        try:
            neg_f1 = 2 * ((neg_precision * neg_recall)/(neg_precision + neg_recall))
        except:
            neg_f1 = 0

        self.accuracy = round(((self.t_pos + self.t_neg)/(self.t_pos + self.t_neg + self.f_pos + self.f_neg))*100, 3)
        precision = round((pos_precision + neg_precision)/2, 5)
        recall = round((pos_recall + neg_recall)/2, 5)
        f1 = 2 * ((precision * recall)/(precision + recall))

        if logs:
            classes_str = "Model, Combined,,,,"
            class_results = ""

            print("Binary Class Evaluation")
            print()
            print(f"True Positive : {self.t_pos}")
            print(f"False Positive : {self.f_pos}")
            print(f"False Negative : {self.f_neg}")
            print(f"True Negative : {self.t_neg}")

            print(f"\nClass {classes[1]} Evaluation")
            print(f"- Precision : {round(pos_precision*100, 3)} %")
            print(f"- Recall : {round(pos_recall*100, 3)} %")
            print(f"- F1 : {round(pos_f1, 5)}")
            classes_str += f"{classes[1]},,,"
            class_results += f"{round(pos_precision*100, 3)}, {round(pos_recall*100, 3)}, {round(pos_f1, 5)}, "

            print(f"\nClass {classes[0]} Evaluation")
            print(f"- Precision : {round(neg_precision*100, 3)} %")
            print(f"- Recall : {round(neg_recall*100, 3)} %")
            print(f"- F1 : {round(neg_f1, 5)}")
            classes_str += f"{classes[0]},,,"
            class_results += f"{round(neg_precision*100, 3)}, {round(neg_recall*100, 3)}, {round(neg_f1, 5)}, "

            print(f"\nCombined Evaluation")
            print(f"- Accuracy : {self.accuracy} %")
            print(f"- Precision : {round(precision*100, 3)} %")
            print(f"- Recall : {round(recall*100, 3)} %")
            print(f"- F1 : {round(f1, 5)}")
            print(f"- Average Confidence : {round((self.confidence/len(self.labels))*100, 2)} %")

            print(classes_str)
            print(f"{self.model_name}, {self.accuracy}, {round(precision*100, 3)}, {round(recall*100, 3)}, {round(f1, 5)}, {class_results}")

    def evaluate_multi_class(self, classes: List[str], logs: bool = True):
        self.labels = self.labels.tolist()
        self.predictions = [pred.tolist() for pred in self.predictions]
        if logs:
            print(f"{len(self.labels)} vs {len(self.predictions)}")

        confusion_matrix = []
        for i in range(len(classes)):
            confusion_matrix.append([])
            for j in range(len(classes)):
                confusion_matrix[i].append(0)

        for i, label in enumerate(self.labels):
            class_pred = self.predictions[i].index(max(self.predictions[i]))
            class_label = label.index(max(label))

            confusion_matrix[class_pred][class_label] += 1

            self.confidence += self.predictions[i][class_pred]

        class_eval = []
        for i, cl in enumerate(classes):
            eval = {
                'label': cl,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
            tp = confusion_matrix[i][i]
            fp = sum([val for val in confusion_matrix[i]]) - tp
            fn = sum([val[i] for val in confusion_matrix]) - tp
            eval['precision'] = 0 if (tp + fp) == 0 else tp/(tp + fp)
            eval['recall'] = 0 if (tp + fn) == 0 else tp/(tp + fn)
            try:
                eval['f1'] = 2 * ((eval['precision'] * eval['recall'])/(eval['precision'] + eval['recall']))
            except:
                eval['f1'] = 0
            class_eval.append(eval)

        accuracy_up = sum([val[i] for i, val in enumerate(confusion_matrix)])
        accuracy_low = sum([sum(val) for val in confusion_matrix])
        self.accuracy = round((accuracy_up/accuracy_low)*100, 3)
        precision = sum([val['precision'] for val in class_eval])/len(classes)
        precision = round(precision, 5)
        recall = sum([val['recall'] for val in class_eval])/len(classes)
        recall = round(recall, 5)
        f1 = 2 * ((precision * recall)/(precision + recall))

        if logs:
            classes_str = "Model, Combined,,,,"
            class_results = ""

            print("Multi Class Evaluation")
            for eval in class_eval:
                classes_str += f"{eval['label']},,,"
                class_results += f"{round(eval['precision']*100, 3)}, {round(eval['recall']*100, 3)}, {round(eval['f1'], 5)}, "

                print(f"\nClass {eval['label']} Evaluation")
                print(f"- Precision : {round(eval['precision']*100, 3)} %")
                print(f"- Recall : {round(eval['recall']*100, 3)} %")
                print(f"- F1 : {round(eval['f1'], 5)}")

            print(f"\nCombined Evaluation")
            print(f"- Accuracy : {self.accuracy} %")
            print(f"- Precision : {round(precision*100, 3)} %")
            print(f"- Recall : {round(recall*100, 3)} %")
            print(f"- F1 : {round(f1, 5)}")
            print(f"\n- Average Confidence : {round((self.confidence/len(self.labels))*100, 2)} %")

            print(classes_str)
            print(f"{self.model_name}, {self.accuracy}, {round(precision*100, 3)}, {round(recall*100, 3)}, {round(f1, 5)}, {class_results}")
