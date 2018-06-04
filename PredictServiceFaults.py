import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

event_type = pd.read_csv("event_type.csv")
log_feature = pd.read_csv("log_feature.csv")
resource_type = pd.read_csv("resource_type.csv")
severity_type = pd.read_csv("severity_type.csv")
train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

train_dataset["source"] = "train"
test_dataset["source"] = "test"

concat_dataset = pd.concat([train_dataset, test_dataset], axis=0, ignore_index=True)
concat_dataset.location = concat_dataset.location.str.split().str.get(1).astype(int)

et_index = event_type.event_type.value_counts(dropna=False)
et_top = et_index[0:15]
event_type = event_type[event_type.event_type.isin(et_top.index)]

#Pivot event_type based on id
event_type = event_type.pivot(index="id", columns="event_type", values="event_type")
event_type = event_type.fillna(0)
event_type["id"] = event_type.index.values

#Event Type
et_X = event_type.iloc[:, 0:15]
et_label_encoder = LabelEncoder()
et_X = et_label_encoder.fit_transform(et_X)


et_one_hot_encoder = OneHotEncoder(categorical_features=list(range(0, len(et_X.columns))))
et_X_encoded = et_one_hot_encoder.fit_transform(et_X).toarray()
et_X = pd.DataFrame(et_X_encoded)
event_type["id"] = event_type[15].astype(int)
del(event_type[15])


#Merge datasets based on column "id"
merged_dataset = concat_dataset.merge(event_type, on="id", how="outer")


#Perform similar steps on log_feature
log_feature["volume"] = np.log(1+log_feature["volume"])
lower_vol = log_feature.volume.mean() - 0.69
upper_vol = log_feature.volume.mean() + 0.69
log_feature = log_feature[np.logical_and(log_feature["volume"] > lower_vol,
                                         log_feature["volume"] < upper_vol)]

lf_perc = pd.DataFrame(round(log_feature.log_feature.value_counts(dropna=False)/
                             sum(log_feature.log_feature.value_counts(dropna=False)),
                             3))
lf_perc_top = lf_perc[lf_perc.log_feature >= 0.02]
log_feature = log_feature[log_feature.log_feature.isin(lf_perc_top.index)]

lg_index = log_feature.log_feature.value_counts(dropna=False)

#pivot
log_feature = log_feature.pivot(index="id", columns="log_feature", values="volume")
log_feature = log_feature.fillna(0)
log_feature["id"] = log_feature.index.values
#Merge datasets based on column "id"
merged_dataset = merged_dataset.merge(log_feature, on="id", how="outer")
merged_dataset = merged_dataset.fillna(0)

#perform similar steps on resource_type
#pivot
resource_type["weight"] = 1
resource_type = resource_type.pivot(index="id", columns="resource_type",
                                    values="weight")
resource_type["id"] = resource_type.index.values
resource_type = resource_type.fillna(0)

del(resource_type["resource_type 9"])

#Merge datasets based on column "id"
merged_dataset = merged_dataset.merge(resource_type, on="id")

#Severity Type
st_label_encoder = LabelEncoder()
severity_type.severity_type = st_label_encoder.fit_transform(severity_type.severity_type)


st_one_hot_encoder = OneHotEncoder(categorical_features=[1])
severity_type_encoded = st_one_hot_encoder.fit_transform(severity_type).toarray()
severity_type = pd.DataFrame(severity_type_encoded)
severity_type["id"] = severity_type[5].astype(int)
del(severity_type[5])



#Merge datasets based on column "id"
merged_dataset = merged_dataset.merge(resource_type, on="id")

train_dataset = merged_dataset[merged_dataset.source == "train"]
test_dataset = merged_dataset[merged_dataset.source == "test"]
del(test_dataset["fault_severity"])
del(train_dataset["source"])
del(test_dataset["source"])

X = train_dataset.iloc[:, 1:]
y = train_dataset.iloc[:, 0]



loc_label_encoder = LabelEncoder()
X.location = loc_label_encoder.fit_transform(X.location)

cat_index = X.columns.get_loc("location")

loc_one_hot_encoder = OneHotEncoder(categorical_features=[cat_index])
location_encoded = loc_one_hot_encoder.fit_transform(X).toarray()
X = pd.DataFrame(location_encoded)

test_loc_label_encoder = LabelEncoder()
test_dataset.location = test_loc_label_encoder.fit_transform(test_dataset.location)

cat_index = test_dataset.columns.get_loc("location")
test_loc_one_hot_encoder = OneHotEncoder(categorical_features=[cat_index])

test_location_encoded = test_loc_one_hot_encoder.fit_transform(test_dataset).toarray()
test_dataset = pd.DataFrame(test_location_encoded)


from xgboost import XGBClassifier
model = XGBClassifier(eta=0.1, max_depth=10, subsample=0.8, colsample_bytree=0.8,
                      )
model.fit(X, y)

# Predicting the Test set results
y_pred = model.predict(test_dataset)
y_pred_df = pd.DataFrame(y_pred)

pred_dict = {}
pred_0 = []
pred_1 = []
pred_2 = []
for index, pred in y_pred_df.iterrows():
    if pred[0] == 0:
        pred_0.append(1)
        pred_1.append(0)
        pred_2.append(0)
    elif pred[0] == 1:
        pred_0.append(0)
        pred_1.append(1)
        pred_2.append(0)
    else:
        pred_0.append(0)
        pred_1.append(0)
        pred_2.append(1)

pred_dict["id"] = test_dataset["id"].values
pred_dict["predict_0"] = pred_0
pred_dict["predict_1"] = pred_1
pred_dict["predict_2"] = pred_2

pred_dict_df = pd.DataFrame(pred_dict)
pred_dict_df.to_csv("output.csv", index=False)