import faiss
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm

def concat_columns(df: pd.DataFrame, columns: list[str]):
    """Concatenate the columns into a new column.

    Args:
        df (pd.DataFrame): The dataframe.
        columns (List[str]): The columns to combine.
    """

    # validate that the columns are of type string
    for col in columns:
        if df[col].dtype != "object":
            raise ValueError(f"Column {col} is not of type object.")
    return df[columns].apply(lambda x: " ".join(x), axis=1)

def semantic_deduplication(
    df: pd.DataFrame,
    required_columns: list[str],
    num_kmeans_clusters: int,
    epsilon: float = 0.99,
    similarity_metric: str = "cosine",
    keep_central: bool = True,
    kmeans_with_cosine_distance: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_batch_size: int = 100,
    use_gpu: bool = False,
):
    """
    Perform semantic deduplication on a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        required_columns (List[str]): The columns to use for deduplication. Will be concatenated in order.
        epsilon (float): The epsilon value to use for semantic deduplication.
            Pairs of items with similarity above epsilon will be considered duplicates.
        num_kmeans_clusters (int): The number of clusters to use in kmeans.
        similarity_metric (str): The similarity metric to use, only "cosine" currently implemented.
        keep_central (bool): Whether to keep the item closest (if True) or farther (if False)
            from the cluster centroid when determining which item to keep.
        kmeans_with_cosine_distance (bool): Whether to use cosine distance for kmeans,
            only False currently implemented.
        model_name (str): The model name to use for embedding.
        embedding_batch_size (int): The batch size to use for embedding.
    """

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_sentences_batch(sentences):
        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        if use_gpu:
            encoded_input = encoded_input.to("cuda")

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )

        return sentence_embeddings

    def embed_sentences(sentences, batch_size=embedding_batch_size):
        """
        Embed a list of sentences using the sentence-transformers model.

        Parameters
        ----------
        sentences : List[str]
            The list of sentences to embed.
        batch_size : int
            The batch size for embedding.

        Returns
        -------
        torch.Tensor
            The sentence embeddings.
        """

        # iterate over the sentences in batches
        sentence_embeddings = []
        for i in tqdm(
            range(0, len(sentences), batch_size),
            dynamic_ncols=True,
            desc="Embedding sentences...",
        ):
            batch = sentences[i : i + batch_size]
            sentence_embeddings.append(embed_sentences_batch(batch))

        return torch.cat(sentence_embeddings)

    def semdedup(cluster, eps=0.95):
        # compute pairwise cosine similarity between cluster items
        pairwise_similarity_matrix = cluster @ cluster.T

        # filter out the diagonal elements
        pairwise_similarity_matrix.fill_diagonal_(0.0)

        # create the upper triangular matrix
        upper_triangular = torch.triu(pairwise_similarity_matrix)

        # create a binary matrix, where 1 means the similarity is above epsilon
        matrix_of_removals = torch.where(upper_triangular > eps, 1, 0)
        # get the indices to remove
        # head is the row, tail is the column
        head_duplicates, tail_duplicates = matrix_of_removals.nonzero(as_tuple=True)

        # get the indices to remove
        indices_to_remove = tail_duplicates.tolist()

        return (
            list(set(indices_to_remove)),
            head_duplicates.tolist(),
            tail_duplicates.tolist(),
        )

    content_col = "dedup_content"
    df[content_col] = concat_columns(df, required_columns)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if use_gpu:
        model = model.to("cuda")

    # embed the content
    embedded_content = embed_sentences(
        df[content_col].tolist(), batch_size=embedding_batch_size
    )

    if use_gpu:
        embedded_content = embedded_content.to("cpu")

    kmeans = faiss.Kmeans(
        d=embedded_content.size(1),
        k=num_kmeans_clusters,
        niter=20,
        verbose=True,
        seed=42,
        spherical=kmeans_with_cosine_distance,  # only true if using cosine distance
        gpu=use_gpu,
    )

    # train the kmeans object
    kmeans.train(embedded_content)
    centroids = kmeans.centroids

    # get the nearest centroid for each data point
    dist_to_cent, nearest_cent = kmeans.index.search(embedded_content, 1)
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(), nearest_cent.squeeze()

    # assign the distance and cluster to the dataframe
    df["distance_to_centroid"] = dist_to_cent
    df["kmeans_cluster"] = nearest_cent

    indices_to_remove = []
    cluster_duplicates_dfs = {}

    for cluster_id in tqdm(
        range(num_kmeans_clusters), desc="Iterating over clusters..."
    ):
        cluster_df = df[df["kmeans_cluster"] == cluster_id]

        # if cluster is empty, skip
        if len(cluster_df) == 0:
            continue

        # get only items from this cluster
        cluster_idxs = cluster_df.index.tolist()
        cluster_embeddings = embedded_content[cluster_idxs]

        # compute the similarity to the centroid
        if similarity_metric == "cosine":
            if kmeans_with_cosine_distance:
                # if cosine distance was used for kmeans clustering, don't recompute
                cluster_dists_to_cent = 1 - cluster_df["distance_to_centroid"]
            else:
                # compute the cosine similarity to the centroid
                cluster_centroid = torch.tensor(centroids[cluster_id])
                sim_to_cent = torch.nn.functional.cosine_similarity(
                    cluster_embeddings, cluster_centroid
                )
                cluster_dists_to_cent = 1 - sim_to_cent
        elif similarity_metric == "l2":
            cluster_dists_to_cent = cluster_df["distance_to_centroid"]

        # sort the cluster items by distance to centroid
        sort_descending = (
            keep_central  # if keep_central is True, sort in descending order
        )
        cluster_sorted = sorted(
            zip(cluster_idxs, cluster_embeddings, cluster_dists_to_cent),
            key=lambda x: x[2],
            reverse=sort_descending,
        )

        # get the sorted indices
        sorted_cluster_idxs = [x[0] for x in cluster_sorted]
        sorted_cluster_embeddings = torch.stack([x[1] for x in cluster_sorted])

        # use semdedup to determine which items to remove
        (
            cluster_indices_to_remove,
            cluster_head_duplicates,
            cluster_tail_duplicates,
        ) = semdedup(sorted_cluster_embeddings, eps=epsilon)

        while cluster_head_duplicates:
            assert len(cluster_head_duplicates) == len(
                cluster_tail_duplicates
            ), "Lengths of head and tail duplicates should be the same."

            # get the first pair of duplicates
            head_idx = cluster_head_duplicates.pop(0)
            tail_idx = cluster_tail_duplicates.pop(0)

            # if the head index is not in the duplicates, create a new dataframe for it
            if sorted_cluster_idxs[head_idx] not in cluster_duplicates_dfs:
                cluster_duplicates_dfs[sorted_cluster_idxs[head_idx]] = pd.DataFrame(
                    columns=df.columns
                )
                cluster_duplicates_dfs[sorted_cluster_idxs[head_idx]].loc[
                    sorted_cluster_idxs[head_idx]
                ] = df.loc[sorted_cluster_idxs[head_idx]]

            # add the tail index to the head duplicates dataframe
            cluster_duplicates_dfs[sorted_cluster_idxs[head_idx]].loc[
                sorted_cluster_idxs[tail_idx]
            ] = df.loc[sorted_cluster_idxs[tail_idx]]

            # remove the tail index if it appears in the head duplicates,
            # prevents duplicates from being counted multiple times
            tail_indxs_in_head = [
                i for i, x in enumerate(cluster_head_duplicates) if x == tail_idx
            ]
            # remove in reverse order so that the indices don't change
            for i in tail_indxs_in_head[::-1]:
                cluster_head_duplicates.pop(i)
                cluster_tail_duplicates.pop(i)

        # convert cluster indices to global indices
        global_indices_to_remove = [
            sorted_cluster_idxs[i] for i in cluster_indices_to_remove
        ]

        indices_to_remove.extend(global_indices_to_remove)

    # # remove the duplicates
    # df = df.drop(indices_to_remove)

    # remove the temporary columns
    df = df.drop(
        columns=["dedup_content", "distance_to_centroid", "kmeans_cluster"], axis=1
    )

    return df, indices_to_remove, cluster_duplicates_dfs