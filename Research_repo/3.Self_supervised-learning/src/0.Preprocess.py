#!/opt/anaconda3/bin/anaconda


from fastai.tabular.all import *
from fastcore.utils import *
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

"""
Args:
    To do
    
Returns:
    emb_xs: a pandas dataframe
    emb_valid_xs : a pandas dataframe
    
"""

df_nn = pd.read_csv('dataset/train.csv', low_memory=False)
df_nn_final = df_nn.drop('id', axis=1)

"""
Categorical embedding
"""

cont,cat = cont_cat_split(df_nn_final, max_card=9000, dep_var='target')
procs_nn = [Categorify, Normalize]
splits = RandomSplitter(seed=23)(df_nn_final)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

to_nn = TabularPandas(df_nn, procs_nn, cat, cont,
                      splits=splits, y_names='target')
dls = to_nn.dataloaders(1024, device = device)

learn = tabular_learner(dls, layers=[500,250], n_out=1)
learn.fit_one_cycle(8, 5e-4)

preds,targs = learn.get_preds()
roc_auc_score(targs, preds)

learn.save('learn8')

# Machine Learning Models
df = pd.read_csv('dataset/train.csv', low_memory=False)
df = df.drop('id', axis=1)
# using the neural net's `cat`, `cont`, and `splits`
procs = [Categorify]
to = TabularPandas(df, procs, cat, cont, 'target', splits = splits)

def rf(xs, y, n_estimators=40, max_samples=130_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf).fit(xs, y)

def auc(m, xs, y):
    preds = m.predict(xs)
    return round(roc_auc_score(y, preds), 3)

# Replacing Nominal variables with Embeddings
learn = learn.load('learn8')

def embed_features(learner, xs):
    """
    learner: fastai Learner used to train the neural net
    xs: DataFrame containing input variables with nominal values defined by their rank.
    ::returns:: a copy of `xs` with embeddings replacing each categorical variable
    """
    xs = xs.copy()
    for i,col in enumerate(learn.dls.cat_names):
        emb = learn.model.embeds[i]
        emb_data = emb(tensor(xs[col], dtype=torch.int64).to(device))
        emb_names = [f'{col}_{j}' for j in range(emb_data.shape[1])]
        feat_df = pd.DataFrame(data=emb_data, index=xs.index, columns=emb_names)
        xs = xs.drop(col, axis=1)
        xs = xs.join(feat_df)
        return xs
    
emb_xs = embed_features(learn, to.train.xs)
emb_valid_xs = embed_features(learn, to.valid.xs)

emb_xs.to_csv('dataset/emb_xs.csv', index=False)
emb_xs.to_csv('dataset/emb_valid_xs.csv', index=False)
df.to_csv('dataset/df.csv', index=False)
