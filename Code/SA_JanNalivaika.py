import math

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC
from tslearn.utils import to_time_series_dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from tslearn.svm import TimeSeriesSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error


def latex_plot(Y, name, step):
    plt.rcParams["figure.figsize"] = [21 / 2, 7.5 / 2]  # set figure size
    plt.ylabel("Force in [kN]")
    plt.xlabel(f'Cycle')
    _, pos_50 = get_damage(Y, 0.5, step)
    _, pos_80 = get_damage(Y, 0.8, step)
    end, pos_100 = get_damage(Y, 1, step)
    plt.title(f"Augmented Load History 5, D = {round(end, 2)}")
    if pos_50 > 0: plt.axvline(x=pos_50, color='green', label='D = 0.5')
    if pos_80 > 0: plt.axvline(x=pos_80, color='orange', label='D = 0.8')
    if pos_100 > 0: plt.axvline(x=pos_100, color='red', label='D = 1')
    plt.plot(Y)
    plt.legend(loc='upper center', bbox_to_anchor=(1.08, 1), fancybox=False, shadow=True)  # love legend right
    #plt.savefig(f"../Latex/IMGs/Verlauf_5_{name}.jpg", bbox_inches='tight', dpi=800)
    plt.close()


def get_damage(force, Damage_cut, step):
    force = abs(force)
    k = 6.33048
    C = 11670367740000000
    D = 0
    Dauerfestigkeit = 56.618
    end = 0
    for idx, elem in enumerate(force):
        if elem > Dauerfestigkeit:
            possible_N = C * (elem ** -k)
            D += step / possible_N
            if D >= Damage_cut and end == 0:
                end = idx
    return D, end


def txt_to_numpy(files):
    if os.path.exists("./Load_sequences_NP"): shutil.rmtree("./Load_sequences_NP")  # if folder exists - delete folder
    os.mkdir("Load_sequences_NP")  # create a new folder

    for idx, file in enumerate(files):  # loop over all textfiles
        print(f"Working on file {idx}")  # making sure its working
        data = [[], [], [], [], [], [], []]  # Empty array to store recorded values

        f = open(file, "r")  # opening file
        file_content = f.readlines()  # Reading line by line
        file_content.pop(0)  # First 2 lines are removed - they contain only text
        file_content.pop(0)

        for element in file_content:  # for every line
            element_sep = element.replace("\n", "").replace(",", ".").split("\t")  # splitting line
            for x in range(7):  # for each element
                data[x].append(element_sep[x])  # saving in array

        data = np.asarray(data).astype(float)  # transforming onto a np array
        fin = np.abs(data[6] - data[3])  # calculating net force
        np.save(f'Load_sequences_NP/{idx}.npy', fin)  # Saving array into folder


def plot_load_sequence():
    if os.path.exists("./Load_sequences_Images"): shutil.rmtree("./Load_sequences_Images")  # if folder exists- delete
    os.mkdir("Load_sequences_Images")  # create new folder

    files = glob.glob("Load_sequences_NP/*")  # path of numpy arrays  - was created by "txt_to_numpy()"

    for idx, element in enumerate(files):  # for every array
        print(f"Plotting Image {idx}")  # Make sure its forking
        arr = np.load(element)  # load array
        plt.rcParams["figure.figsize"] = [21 / 2, 7.5 / 2]  # set figure size

        _, a = get_damage(arr, 0.5, 1)  # calculate position of Damage = 0.5
        _, b = get_damage(arr, 0.8, 1)  # calculate position of Damage = 0.8
        end, c = get_damage(arr, 1, 1)  # calculate position of Damage = 1 AND absolute damage

        plt.plot(arr)  # Plot array

        if a > 0: plt.axvline(x=a, color='green', label='D = 0.5')  # if sequence reached D = 0.5 plot line
        if b > 0: plt.axvline(x=b, color='orange', label='D = 0.8')  # if sequence reached D = 0.8 plot line
        if c > 0: plt.axvline(x=c, color='red', label='D = 1')  # if sequence reached D = 1 plot line

        plt.ylabel("Force in [kN]")  # Y Label
        plt.xlabel('Cycle')  # X Label
        plt.title(f"Load History: {idx}, Failure at D = {round(end, 2)}")  # Title
        plt.legend(loc='upper center', bbox_to_anchor=(1.08, 1), fancybox=False, shadow=True)  # love legend right
        plt.savefig(f"Load_sequences_Images/Verlauf_{idx}.jpg", bbox_inches='tight', dpi=800)  # Save image
        plt.close()  # Close figure


def data_augmentation(stepper):
    draw = False  # drawing all files

    augmented_counter = 0
    class1_counter = 0
    class2_counter = 0
    class3_counter = 0
    total_l = 0
    min_l = math.inf
    max_l = 0

    # Making Folders for np-files
    if os.path.exists("./Load_sequences_Augmented"): shutil.rmtree("./Load_sequences_Augmented")
    os.mkdir("Load_sequences_Augmented")

    # Making folder for Images
    if os.path.exists("./Load_sequences_Augmented_Images"): shutil.rmtree("./Load_sequences_Augmented_Images")
    if draw: os.mkdir("Load_sequences_Augmented_Images")

    # Making folder for np-files where instance of a file wer not seen in training - validation set
    if os.path.exists("./Load_sequences_Unseen"): shutil.rmtree("./Load_sequences_Unseen")
    os.mkdir("Load_sequences_Unseen")

    files = glob.glob("Load_sequences_NP/*")

    # ignoring the validation set for now
    random_file_nr = random.randint(0, len(files))
    random_file_nr = -1

    for idx, file in enumerate(files):
        print(f"Augmenting file {idx}")
        plt.rcParams["figure.figsize"] = [21 / 2, 7.5 / 2]  # set figure size
        for x in range(25):
            Y = np.load(file)
            original_dmg, _ = get_damage(Y, 1, 1)

            noise = np.random.normal(0, 1, len(Y))  # Add random noise
            Y += noise
            # if idx == 5: latex_plot(Y,"noise",1)

            shift = np.random.rand(1) * 0.5
            Y += shift
            # if idx == 5: latex_plot(Y, "shift",1)

            for x in range(3):
                start = random.randint(0, len(Y))
                length = int(len(Y) / 100)
                Y[start:length + start] *= random.uniform(0.5, 1.5)
            # if idx == 5: latex_plot(Y, "shift_partial",1)

            for x in range(3):
                start = random.randint(0, len(Y))
                length = int(len(Y) / 100)
                end = min(start + length, len(Y))
                Y = np.delete(Y, list(range(start, end)))
            # if idx == 5: latex_plot(Y, "coutout",1)

            gerade = np.linspace(random.uniform(-0.2, 0), random.uniform(0, 0.2), len(Y))
            if np.random.randint(2) == 0: gerade = -gerade
            Y += gerade
            # if idx == 5: latex_plot(Y, "tilt",1)

            augmented_dmg, _ = get_damage(Y, 1, 1)
            ratio = augmented_dmg / original_dmg


            if 0.9 < ratio < 1.1:

                Y = Y[0::stepper]  # slecting every nth step

                reduced_dmg, _ = get_damage(Y, 1, stepper)
                ratio = reduced_dmg / original_dmg

                if 0.9 < ratio < 1.1:
                    # if idx == 5:
                    # latex_plot(Y, "reduced",stepper)

                    # assigning a class
                    if reduced_dmg < 0.90:
                        label = -1
                        class1_counter += 1
                    elif reduced_dmg > 1.10:
                        label = 1
                        class3_counter += 1
                    else:
                        label = 0
                        class2_counter += 1

                    if idx == random_file_nr:
                        # creating validation set
                        np.save(f"Load_sequences_Unseen/{label}_{augmented_counter}.npy", Y)
                    else:
                        np.save(f"Load_sequences_Augmented/{label}_{augmented_counter}.npy", Y)

                    augmented_counter += 1
                    total_l += len(Y)
                    if len(Y) > max_l: max_l = len(Y)
                    if len(Y) < min_l: min_l = len(Y)

                    # Plotting all DA-sequences in one plot
                    if draw:
                        _, pos_50 = get_damage(Y, 0.5, stepper)
                        _, pos_80 = get_damage(Y, 0.8, stepper)
                        _, pos_100 = get_damage(Y, 1, stepper)

                        if pos_50 > 0: plt.axvline(x=pos_50, color='green', label='D = 0.5', lw=0.5, linestyle='dashed')
                        if pos_80 > 0: plt.axvline(x=pos_80, color='orange', label='D = 0.8', lw=0.5, linestyle='dashed')
                        if pos_100 > 0: plt.axvline(x=pos_100, color='red', label='D = 1', lw=0.5, linestyle='dashed')

                        plt.plot(Y)
        if draw:
            plt.ylabel("Force in [kN]")
            plt.xlabel(f'Cycle * {stepper}')
            plt.title(f"load sequnece {idx}, Augmented Data, D_original = {round(original_dmg, 2)}")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(1.08, 1), fancybox=False,
                       shadow=True)  # love legend right
            plt.savefig(f"Load_sequences_Augmented_Images/Augmented_{idx}.png", bbox_inches='tight', dpi=800)
        plt.close()

    print(class1_counter, class2_counter, class3_counter)
    print(f"Average lenght = {total_l/augmented_counter}")
    print(f"Max and Min lenght = {max_l}, {min_l}")


def load_data_classifier(D, stepper):
    files = glob.glob("Load_sequences_Augmented/*")
    big_arr = []
    labels = []
    print("Loading files")
    for file in files:
        arr = np.load(file)
        _, pos = get_damage(arr, D, stepper)
        # only include array in trainig set if damage sum D is higher that total damage of the load sequence
        if pos >  0:
            arr = arr[:pos] # Cut array at damage sum D of load sequences in testing set
            big_arr.append(arr)
            label = int((file.split("_")[2]).split("\\")[-1])
            labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(big_arr, labels, test_size=0.2, random_state=42, shuffle=True,
                                                        stratify=labels)  #
    # print(np.sum(np.asarray(y_train) == -1), np.sum(np.asarray(y_train) == 0), np.sum(np.asarray(y_train) == 1))

    return X_train, X_test, y_train, y_test


def train_test_classifier(X, x, Y, y, model):
    print("Stating Training")
    longest_list = 0
    Y = np.asarray(Y)
    if model == 1 or model == 2:
        if model == 1: clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw", n_jobs=-1)
        if model == 2: clf = TimeSeriesSVC(C=1.0, kernel="gak")

        IN = to_time_series_dataset(X)
        clf = clf.fit(IN, Y)

        print("Starting Testing")
        IN = to_time_series_dataset(x)
        out = clf.predict(IN)

    else:
        if model == 3: clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        if model == 4: clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                                        random_state=0)
        if model == 5: clf = RandomForestClassifier(max_depth=20, random_state=0)
        if model == 6:
            clf = XGBClassifier(n_estimators=20, max_depth=20, learning_rate=1, objective='binary:logistic')
            Y += 1
        if model == 7: clf = MLPClassifier(random_state=1, max_iter=300)
        if model == 8: clf = GaussianNB()
        if model == 9: clf = QuadraticDiscriminantAnalysis()
        if model == 10: clf = KNeighborsClassifier(n_neighbors=3)

        longest_list = max(max(len(elem) for elem in X), max(len(elem) for elem in x))
        X = [np.insert(elem, 0, np.zeros(longest_list - len(elem)), axis=0) for elem in X]

        clf.fit(np.asarray(X), Y)
        print("Starting Testing")
        IN = [np.insert(elem, 0, np.zeros(longest_list - len(elem)), axis=0) for elem in x]
        out = clf.predict(IN)

    correct = np.sum(out == y) / len(y)
    print(correct)

    return clf, longest_list, correct


def unseen_data_validation(clf, model, l):
    files = glob.glob("Load_sequences_Unseen/*")
    big_arr = []
    labels = []
    for file in files:
        arr = np.load(file)
        big_arr.append(arr)
        label = int((file.split("_")[2]).split("\\")[-1])
        labels.append(label)

    if model == 1 or model == 2:
        X = to_time_series_dataset(big_arr)
    else:
        X = [np.insert(elem, 0, np.zeros(l - len(elem)), axis=0) for elem in big_arr]

    out = clf.predict(X)
    correct = np.sum(out == label) / len(labels)
    print(correct)
    return correct


def load_data_regressor(D, s, c):
    files = glob.glob(f"Load_sequences_Augmented/{c}*")
    big_arr = []
    labels = []
    print("Loading files")
    for file in files:
        arr = np.load(file)
        _, pos = get_damage(arr, D, s)
        label_function = np.linspace(0, 1, len(arr))
        for x in range(10):
            end = random.randint(1, len(arr) - 1)
            arr_short = arr[:end]
            big_arr.append(arr_short)
            label = label_function[end]
            labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(big_arr, labels, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test


def train_test_regressor(X, x, Y, y, model):
    print("Stating Training")

    longest_list = max(max(len(elem) for elem in X), max(len(elem) for elem in x))
    X = [np.insert(elem, 0, np.zeros(longest_list - len(elem)), axis=0) for elem in X]

    if model == 1: reg = AdaBoostRegressor(random_state=0, n_estimators=100)
    if model == 2: reg = RandomForestRegressor(max_depth=20, random_state=0)

    reg = reg.fit(X, Y)

    x = [np.insert(elem, 0, np.zeros(longest_list - len(elem)), axis=0) for elem in x]
    out = reg.predict(x)

    error1 = mean_squared_error(y, out, squared=False)

    return error1


def barplot(s):
    plt.rcParams["figure.figsize"] = [21 / 2, 9 / 2]
    fig = plt.figure()

    x = ["KNeighborsTimeSeriesClassifier",
         "TimeSeriesSVC",
         "AdaBoostClassifier",
         "GradientBoostingClassifier",
         "RandomForestClassifier",
         "XGBoost",
         "Neural Net",
         "GaussianNB",
         "QuadraticDiscriminantAnalysis",
         "KNeighborsClassifier"]
    plt.bar(x, s)
    plt.xticks(rotation=45, ha='right')
    plt.title("Average performance of Classifiers")
    plt.savefig(f"Average_performance_CLASS.png", bbox_inches='tight', dpi=800)


def barplot_regression(M):
    plt.rcParams["figure.figsize"] = [21 / 2, 9 / 2]
    barWidth = 0.3

    bars1 = M[0]

    # Choose the height of the cyan bars
    bars2 = M[1]

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Create blue bars
    plt.bar(r1, bars1, width=barWidth, edgecolor='black', capsize=7, label='AdaBoostRegressor')

    # Create cyan bars
    plt.bar(r2, bars2, width=barWidth, edgecolor='black', capsize=7, label='RandomForestRegressor')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Class -1', 'Class 0', 'Class 1'])
    plt.ylabel('Error')
    plt.legend()
    plt.title("Average performance of Regressors")
    plt.savefig(f"Average_performance_REG.png", bbox_inches='tight', dpi=800)
    # Show graphic
    plt.close()
    # plt.show()


if __name__ == "__main__":
    """
    SELECT A MODEL FOR CLASSIFICATION
    
    1) KNeighborsTimeSeriesClassifier
    2) TimeSeriesSVC
    3) AdaBoostClassifier
    4) GradientBoostingClassifier
    5) RandomForestClassifier
    6) XGBoost
    7) Neural Net
    8) GaussianNB
    9) QuadraticDiscriminantAnalysis
    10) KNeighborsClassifier
    """
    selected_model_classification = 1  # Choose one model for testing
    D_cutoff = 0.6  # Select a cutoff point for Damage sum D
    files = glob.glob("Load_sequences_raw/*.txt")  # path to files
    stepper = 300  # step size for Dim Reduction
    score = np.zeros(10)  # value to keep track of score if model is tested over multiple files

    txt_to_numpy(files)  # Converting txt files to numpy arrays and storing in a new folder
    plot_load_sequence()  # Plotting all load sequences
    data_augmentation(stepper)  # Performing DA on all sequences


    loops = 25
    for model in range(10):
        selected_model_classification = model + 1
        for loop in range(loops):
            data_augmentation(stepper)
            X_train_c, X_test_c, y_train_c, y_test_c = load_data_classifier(D_cutoff, stepper)
            classifier, length, correct = train_test_classifier(X_train_c, X_test_c, y_train_c, y_test_c,
                                                                selected_model_classification)
            score[model] += correct / loops
            print(score)
            barplot(score)

    """Use this only if validation set is present"""
    # score_unseen = unseen_data_validation(classifier, selected_model, length)/loops


    """
    SELECT A MODEL FOR REGRESSION
    1) AdaBoostRegressor
    2) RandomForestRegressor
    """
    score_matrix = np.zeros((2, 3))

    for loop in range(loops):
        for model in range(2):
            data_augmentation(stepper)
            for cat in range(3):
                predicted_class = cat - 1
                X_train_r, X_test_r, y_train_r, y_test_r = load_data_regressor(D_cutoff, stepper, predicted_class)
                selected_model_regression = model + 1
                score = train_test_regressor(X_train_r, X_test_r, y_train_r, y_test_r, selected_model_regression)
                score_matrix[model, cat] += score / loops
    print(score_matrix)
    barplot_regression(score_matrix)
