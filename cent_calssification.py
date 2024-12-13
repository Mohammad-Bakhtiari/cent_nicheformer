import math
import os
import mygene
import pandas as pd
import numpy as np
import anndata
import pyarrow.parquet as pq
import pyarrow
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
import numba
from tqdm import tqdm
from nicheformer.models._nicheformer import Nicheformer
from nicheformer.config_files._config_fine_tune import hp_config as config
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from nicheformer.data.datamodules import MerlinDataModuleDistributed
from nicheformer.models._fine_tune_model import FineTuningModel
import argparse

parser = argparse.ArgumentParser(description='Fine-tune Nicheformer model')
parser.add_argument("--dataset_name", type=str, help="Dataset name")
parser.add_argument("--data_path", type=str, help="Path to data directory")
parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")
parser.add_argument("--cell_key", type=str, help="Cell type key", default="Celltype")
parser.add_argument("--batch_key", type=str, help="Batch key", default="batch")


# Dictionary mappings
modality_dict = {'dissociated': 3}
specie_dict = {'human': 5, 'Homo sapiens': 5}
technology_dict = {"10x 3' v3": 11}


def sf_normalize(X):
    """
    Normalize the matrix rows to a fixed scaling factor of 10,000.

    Args:
        X (array or sparse matrix): Gene expression matrix.
    Returns:
        array or sparse matrix: Normalized gene expression matrix.
    """
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    counts += counts == 0.  # Avoid division by zero
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x, max_seq_len=-1, aux_tokens=30):
    """
    Internal function to tokenize data for each cell.

    Args:
        x (np.array): Normalized gene expression matrix.
        max_seq_len (int): Maximum sequence length.
        aux_tokens (int): Reserved tokens for padding or special purposes.
    Returns:
        np.array: Tokenized data.
    """
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices += aux_tokens
        scores = np.zeros(max_seq_len, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        scores_final[i, :] = scores
    return scores_final


def tokenize_data(x, median_counts_per_gene, max_seq_len=4096):
    """
    Tokenize the gene expression matrix.

    Args:
        x (np.array): Gene expression matrix.
        median_counts_per_gene (np.array): Median counts per gene for normalization.
        max_seq_len (int): Maximum sequence length.
    Returns:
        np.array: Tokenized data.
    """
    if issparse(x):
        x = x.toarray()
    x = np.nan_to_num(x)
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))
    return _sub_tokenize_data(out, max_seq_len).astype('i4')


def split_reference_into_train_val(reference, val_fraction=0.1, random_seed=42):
    """
    Split reference data into training and validation sets.

    Args:
        reference (AnnData): Reference dataset.
        val_fraction (float): Fraction of data for validation.
        random_seed (int): Random seed for reproducibility.
    Returns:
        (AnnData, AnnData): Training and validation datasets.
    """
    np.random.seed(random_seed)
    total_obs = reference.n_obs
    indices = np.arange(total_obs)
    np.random.shuffle(indices)

    val_size = int(val_fraction * total_obs)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return reference[train_indices].copy(), reference[val_indices].copy()


def tokenize(adata, mean_values, file_path, filename):
    """
    Tokenize and save the data in Parquet format.

    Args:
        adata (AnnData): Dataset to tokenize.
        mean_values (np.array): Median gene counts for normalization.
        file_path (str): Path to save tokenized data.
        filename (str): Filename prefix for saving.
    """

    # Dropping the index as the original index can create issues
    adata.obs.reset_index(drop=True, inplace=True)
    # Writing the data
    adata.write(f"{file_path}/{filename}_ready_to_tokenize.h5ad")
    obs_data = adata.obs
    print('n_obs: ', obs_data.shape[0])
    N_BATCHES = math.ceil(obs_data.shape[0] / 10_000)
    print('N_BATCHES: ', N_BATCHES)
    batch_indices = np.array_split(obs_data.index, N_BATCHES)
    chunk_len = len(batch_indices[0])
    print('chunk_len: ', chunk_len)
    obs_data = obs_data.reset_index().rename(columns={'index': 'idx'})
    obs_data['idx'] = obs_data['idx'].astype('i8')
    for batch in tqdm(range(N_BATCHES)):
        obs_tokens = obs_data.iloc[batch * chunk_len:chunk_len * (batch + 1)].copy()
        tokenized = tokenize_data(adata.X[batch * chunk_len:chunk_len * (batch + 1)], mean_values, 4096)

        obs_tokens = obs_tokens[['assay', 'specie', 'modality', 'idx']]
        # Concatenate dataframes
        obs_tokens['X'] = [tokenized[i, :] for i in range(tokenized.shape[0])]
        # Mix spatial and dissociate data
        obs_tokens = obs_tokens.sample(frac=1)
        obs_tokens['X'] = obs_tokens['X'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        for col in obs_tokens.columns:
            try:
                obs_tokens[[col]].to_json(orient='records')
            except Exception as e:
                print(f"Column '{col}' is not serializable: {e}")
        obs_tokens['assay'] = obs_tokens['assay'].astype('int')
        obs_tokens['specie'] = obs_tokens['specie'].astype('int')
        for col in obs_tokens.columns:
            try:
                pyarrow.Table.from_pandas(obs_tokens[[col]])
            except Exception as e:
                print(f"Column '{col}' failed with error: {e}")
        # Convert to PyArrow table
        total_table = pyarrow.Table.from_pandas(obs_tokens)

        if not os.path.exists(f"{file_path}/{filename}"):
            os.makedirs(f"{file_path}/{filename}")
        # Save as Parquet
        pq.write_table(total_table, f"{file_path}/{filename}/tokens-{batch}.parquet",
                       row_group_size=1024)


def gene_to_ensemble_id(adata, csv_file):
    """
    Maps gene names to Ensembl IDs, handling un-hit and multi-hit genes.
    If there are multi-hit genes, the resolution is read from a CSV file.

    Args:
        adata: AnnData object containing gene names.
        csv_file: CSV file to resolve multi-hit genes. Defaults to 'multi_hit_resolutions.csv'.

    Returns:
        A dictionary mapping gene names to Ensembl IDs.
    """
    # Initialize variables
    gene_names = adata.var.index.tolist()
    gene_mapping = {}
    unhit_genes = []
    multi_hit_genes = {}

    # Initialize MyGeneInfo
    mg = mygene.MyGeneInfo()

    # Query primary symbols
    primary_results = mg.querymany(
        gene_names, scopes='symbol', fields='ensembl.gene', species='human'
    )

    # Process results
    for result in primary_results:
        query_gene = result.get('query')
        if result.get('notfound'):
            unhit_genes.append(query_gene)
        else:
            ensembl_id = result.get('ensembl', {})
            if isinstance(ensembl_id, list):
                # Multi-hit gene
                multi_hit_genes[query_gene] = ensembl_id
            else:
                gene_mapping[query_gene] = ensembl_id.get('gene')

    # Handle un-hit genes by querying aliases
    if unhit_genes:
        alias_results = mg.querymany(
            unhit_genes, scopes='alias', fields='ensembl.gene', species='human'
        )
        unhit_genes = []  # Reset unhit genes
        for result in alias_results:
            query_gene = result.get('query')
            if result.get('notfound'):
                unhit_genes.append(query_gene)
            else:
                ensembl_id = result.get('ensembl', {})
                if isinstance(ensembl_id, list):
                    # Multi-hit gene
                    multi_hit_genes[query_gene] = ensembl_id
                else:
                    gene_mapping[query_gene] = ensembl_id.get('gene')

    # Handle multi-hit genes using the CSV file
    if multi_hit_genes:
        if os.path.exists(csv_file):
            # Load resolutions from the CSV file
            multi_hit_resolutions = pd.read_csv(csv_file)
            for _, row in multi_hit_resolutions.iterrows():
                gene_name = row['gene_symbbol_by_gene_name']
                ensembl_id = row['ens_id']
                if gene_name in multi_hit_genes:
                    gene_mapping[gene_name] = ensembl_id
                    del multi_hit_genes[gene_name]
            if multi_hit_genes:
                print("Remaining multi-hit genes:", multi_hit_genes)
            if unhit_genes:
                print("Un-hit genes:", unhit_genes)
            if not unhit_genes and not multi_hit_genes:
                print("All genes resolved.")
        else:
            raise FileNotFoundError(
                f"CSV file '{csv_file}' for resolving multi-hit genes not found."
            )

    # Log any remaining multi-hit or un-hit genes
    if unhit_genes or multi_hit_genes:
        if unhit_genes:
            print("Un-hit genes:", unhit_genes)
        if multi_hit_genes:
            print("Remaining multi-hit genes:", multi_hit_genes)

    return gene_mapping

def get_common_genes(adata):
    kept = []

    for i in range(len(adata.var.index)):
        if adata.var.index[i] in adata.var.index:
            kept.append(i)
    common_genes = adata.var.iloc[kept].index
    return kept, common_genes

def set_obs(adata, cell_key, celltypes, split_value):
    adata.obs = adata.obs[
        ['assay', 'organism', 'nicheformer_split', 'batch', cell_key]
    ]
    adata.obs['modality'] = 'dissociated'
    adata.obs['nicheformer_split'] = split_value
    adata.obs['specie'] = adata.obs['organism']

    # Replace missing or default values
    # Handle missing values for 'assay'
    if 'assay' in adata.obs.columns:
        if adata.obs['assay'].dtype.name == 'category':
            # Add the new category if it's categorical
            adata.obs['assay'] = adata.obs['assay'].cat.add_categories(["10x 3' v3"])
        adata.obs['assay'].fillna("10x 3' v3", inplace=True)
    else:
        adata.obs['assay'] = "10x 3' v3"

    # Handle missing values for 'organism'
    if 'organism' in adata.obs.columns:
        if adata.obs['organism'].dtype.name == 'category':
            # Add the new category if it's categorical
            adata.obs['organism'] = adata.obs['organism'].cat.add_categories(["Homo sapiens"])
        adata.obs['organism'].fillna("Homo sapiens", inplace=True)
    else:
        adata.obs['organism'] = "Homo sapiens"
    if 'specie' in adata.obs.columns:
        if adata.obs['specie'].dtype.name == 'category':
            adata.obs['specie'] = adata.obs['specie'].cat.add_categories(["human"])
        adata.obs['specie'].fillna("human", inplace=True)
    else:
        adata.obs['specie'] = "human"
    # Replace string values with corresponding dictionary values
    adata.obs.replace({'specie': specie_dict}, inplace=True)
    adata.obs.replace({'modality': modality_dict}, inplace=True)
    adata.obs.replace({'assay': technology_dict}, inplace=True)
    celltype_dict = {celltype: idx for idx, celltype in enumerate(celltypes)}

    # Map cell types to indices in both reference and query datasets
    adata.obs['cell_type_idx'] = adata.obs[cell_key].map(celltype_dict)
    log_obs(adata)

def log_obs(adata):
    print(adata.obs.keys())
    try:
        print("Assay:",adata.obs['assay'].unique())
    except:
        print("assay not found")
    try:
        print("Speice:", adata.obs['specie'].unique())
    except:
        print("specie not found")
    try:
        print("Modality:", adata.obs['modality'].unique())
    except:
        print("modality not found")
    try:
        print("nicheformer_split:", adata.obs['nicheformer_split'].unique())
    except:
        print("nicheformer_split not found")
    try:
        print("Organism", adata.obs['organism'].unique())
    except:
        print("organism not found")


def prep(ds_name, data_path):
    """
    Prepare datasets for training, validation, and testing.

    Args:
        ds_name (str): Dataset name.
    """
    query = anndata.read_h5ad(f"{data_path}/{ds_name}/query.h5ad")
    reference = anndata.read_h5ad(f"{data_path}/{ds_name}/reference_refined.h5ad")
    train_reference, val_reference = split_reference_into_train_val(reference, val_fraction=0.1)
    if reference.obs[args.cell_key].dtype == "category":
        celltypes = reference.obs[args.cell_key].cat.categories.values
    else:
        celltypes = reference.obs[args.cell_key].unique()
    model = anndata.read_h5ad(f"{data_path}/model_means/model.h5ad")

    def order_genes(adata, model_adata):
        gene_map = gene_to_ensemble_id(adata, csv_file=f"{data_path}/{ds_name}/final_{ds_name}_multihit_gene_map.csv")
        adata.var['ensembl'] = adata.var.index.map(gene_map)
        adata.var.set_index('ensembl', inplace=True)
        adata = anndata.concat([model_adata, adata], join='outer', axis=0)
        adata_output = adata[1:].copy()
        adata_output = adata_output[:, model_adata.var.index]
        return adata_output

    train_reference = order_genes(train_reference, model)
    val_reference = order_genes(val_reference, model)
    query = order_genes(query, model)

    kept, common_genes = get_common_genes(train_reference)
    set_obs(train_reference, args.cell_key, celltypes, split_value="train")
    set_obs(val_reference, args.cell_key, celltypes, split_value="val")
    set_obs(query, args.cell_key, celltypes, split_value="test")

    dataset_dir_path = f"{data_path}/{ds_name}"
    os.makedirs(dataset_dir_path, exist_ok=True)

    def apply_tech_mean(adata, model_means_path):
        dissociated_mean = np.load(f"{model_means_path}/dissociated_mean_script.npy")
        dissociated_mean = np.nan_to_num(dissociated_mean)
        dissociated_mean = dissociated_mean[:adata.shape[1]]
        return adata, dissociated_mean

    train_reference, ds_mean = apply_tech_mean(train_reference, f"{data_path}/model_means")
    tokenize(train_reference, ds_mean, dataset_dir_path, filename="train")

    val_reference, ds_mean = apply_tech_mean(val_reference, f"{data_path}/model_means")
    tokenize(val_reference, ds_mean, dataset_dir_path, filename="val")

    query, ds_mean = apply_tech_mean(query, f"{data_path}/model_means")
    tokenize(query, ds_mean, dataset_dir_path, filename="test")


def finetune(ds_path, checkpoint_path):
    """
    Finetune the Nicheformer model using tokenized data.
    """
    model = Nicheformer.load_from_checkpoint(checkpoint_path=checkpoint_path)
    fine_tune_model = FineTuningModel(
        backbone=model,
        freeze=config['freeze'],
        extract_layers=config['extract_layers'],
        function_layers=config['function_layers'],
        reinit_layers=config['reinit_layers'],
        extractor=config['extractor'],
        baseline=config['baseline'],
        warmup=config['warmup'],
        lr=config['lr'],
        max_epochs=config['max_epochs'],
        supervised_task=config['supervised_task'],
        regress_distribution=config['regress_distribution'],
        pool=config['pool'],
        dim_prediction=config['dim_prediction'],
        n_classes=config['n_classes'],
        predict_density=config['predict_density'],
        ignore_zeros=config['ignore_zeros'],
        organ=config['organ'],
        label=config['label']
    )

    checkpoint_callback = ModelCheckpoint(dirpath="/fine_tuned_models",
                                          every_n_train_steps=5000,
                                          monitor='fine_tuning_classification', save_top_k=-1)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=config['max_epochs'],
        devices=-1,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback, lr_monitor],
        precision='bf16-mixed',
        gradient_clip_val=1,
        accumulate_grad_batches=10
    )

    module = MerlinDataModuleDistributed(
        path=ds_path,
        columns=['assay', 'specie', 'modality', 'idx', 'X'],
        batch_size=config['batch_size'],
        world_size=1,
        splits=True
    )

    trainer.fit(model=fine_tune_model, datamodule=module)


if __name__ == '__main__':
    args = parser.parse_args()
    # You can skip the prep step if you have already tokenized the data
    # prep(args.dataset_name, args.data_path)
    finetune(f"{args.data_path}/{args.dataset_name}", args.checkpoint_path)
