# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

It is a RandomForestClassifier from scikit-learn library used for classification problem. All the hyperparameters are default parameters of the classifier.

## Intended Use

The model is used for classifying employees' salary into <=50K and >50K, based on employees's information.

Users can apply this model to their information about the employees in the predefined format and get the prediction for the salary type.

## Training Data

Publicly available Census Bureau dataset is used for training and evaluating the model.

The dataset contains a good amount of examples and lots of features, hence there are enough data to train a good perfoming model.

For both training and evaluation, categorical features of the data are encoded using OneHotEncoder and the target is transformed using LabelBinarizer

## Evaluation Data

The original dataset is first preprocessed and then split into training and evaluation data with evaluation data size of 20%



## Metrics

Please include the metrics used and your model's performance on those metrics. 3 metrics were used for evaluating the model's performance: precision, recall, and fbeta. The model achieves the following result:

- precision: 0.759
- recall: 0.643
- fbeta: 0.696

## Ethical Considerations

The misuse of these census information can cause consequences to the lives of people surveyed and (possibly) other people in some related populations

## Caveats and Recommendations

The model was trained on data of people mostly from the USA. Hence it is not recommended to use the model to predict the salary type for people from other regions in the world, which might have very different feature distributions