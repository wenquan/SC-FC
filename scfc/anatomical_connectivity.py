"""
Turner, Mann, Clandinin: structural connectivity utils and functions.

https://github.com/mhturner/SC-FC
mhturner@stanford.edu

References:
https://connectome-neuprint.github.io/neuprint-python/docs/index.html
https://github.com/connectome-neuprint/neuprint-python
"""

import numpy as np
import pandas as pd
import os
from neuprint import (fetch_neurons, fetch_adjacencies, NeuronCriteria)
from scipy.sparse import csr_matrix

from . import bridge


def getAtlasConnectivity(include_inds, name_list, atlas_id, metric='cellcount'):
    """
    Load .csv of region-to-region structural connectivity, computed from hemibrain_2_atlas.r.

    :include_inds: list of ROI number IDs to select
    :name_list: associated list of ROI names
    :atlas_id:
    :metric: 'cellcount', 'tbar', 'weighted_tbar'
    """
    data_dir = bridge.getUserConfiguration()['data_dir']
    if atlas_id == 'branson':
        cellcount_full = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_branson_{}_matrix.csv'.format(metric)), header=0).to_numpy()[:, 1:]
        cellcount_full = pd.DataFrame(data=cellcount_full, index=np.arange(1, 1000), columns=np.arange(1, 1000))
    elif atlas_id == 'ito':
        cellcount_full = pd.read_csv(os.path.join(data_dir, 'hemi_2_atlas', 'JRC2018_ito_{}_matrix.csv'.format(metric)), header=0).to_numpy()[:, 1:]
        cellcount_full = pd.DataFrame(data=cellcount_full, index=np.arange(1, 87), columns=np.arange(1, 87))

    # filter and sort cellcount_full by include_inds
    cellcount_filtered = pd.DataFrame(data=np.zeros((len(include_inds), len(include_inds))), index=name_list, columns=name_list)

    for s_ind, src in enumerate(include_inds):
        for t_ind, trg in enumerate(include_inds):
            cellcount_filtered.iloc[s_ind, t_ind] = cellcount_full.loc[src, trg]

    return cellcount_filtered


def getRoiCompleteness(neuprint_client, name_list):
    """
    Return roi completness measures for Ito atlas regions.

    neuprint_client
    name_list: list of Ito regions to return completeness for
    """
    # How many synapses belong to traced neurons
    completeness_neuprint = neuprint_client.fetch_roi_completeness()
    completeness_neuprint.index = completeness_neuprint['roi']

    completeness = np.zeros((len(name_list), 2))
    for r_ind, roi in enumerate(name_list):
        current_rois = completeness_neuprint.loc[bridge.ito_to_neuprint(roi), :]
        completeness[r_ind, 0] = current_rois['roipre'].sum() / current_rois['totalpre'].sum()
        completeness[r_ind, 1] = current_rois['roipost'].sum() / current_rois['totalpost'].sum()

    roi_completeness = pd.DataFrame(data=completeness, index=name_list, columns=['frac_pre', 'frac_post'])
    roi_completeness['completeness'] = roi_completeness['frac_pre'] * roi_completeness['frac_post']

    return roi_completeness


def computeConnectivityMatrix(neuprint_client, mapping):
    """
    Compute region connectivity matrix from neuprint tags, for various metrics.

    neuprint_client
    mapping: mapping dict to bridge hemibrain regions to atlas regions
    """
    rois = list(mapping.keys())
    rois.sort()

    WeakConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    MediumConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    StrongConnections = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    Connectivity = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    WeightedSynapseNumber = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)
    TBars = pd.DataFrame(data=np.zeros((len(rois), len(rois))), index=rois, columns=rois)

    body_ids = [] # keep list of all cells connecting among regions

    for roi_source in rois:
        for roi_target in rois:
            sources = mapping[roi_source]
            targets = mapping[roi_target]

            weak_neurons = 0
            medium_neurons = 0
            strong_neurons = 0
            summed_connectivity = 0
            weighted_synapse_number = 0
            tbars = 0
            for s_ind, sour in enumerate(sources): # this multiple sources/targets is necessary for collapsing rois based on mapping
                for targ in targets:
                    Neur, Syn = fetch_neurons(NeuronCriteria(inputRois=sour, outputRois=targ, status='Traced', cropped=False)) # only take uncropped neurons

                    outputs_in_targ = np.array([x[targ]['pre'] for x in Neur.roiInfo]) # neurons with Tbar output in target
                    inputs_in_sour = np.array([x[sour]['post'] for x in Neur.roiInfo]) # neuron with PSD input in source

                    n_weak = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour<3))
                    n_medium = np.sum(np.logical_and(outputs_in_targ>0, np.logical_and(inputs_in_sour>=3, inputs_in_sour<10)))
                    n_strong = np.sum(np.logical_and(outputs_in_targ>0, inputs_in_sour>=10))

                    # Connection strength for each cell := sqrt(input PSDs in source x output tbars in target)
                    conn_strengths = [np.sqrt(x[targ]['pre'] * x[sour]['post']) for x in Neur.roiInfo]

                    # weighted synapses, for each cell going from sour -> targ:
                    #       := output tbars (presynapses) in targ * (input (post)synapses in sour)/(total (post)synapses onto that cell)
                    weighted_synapses = [Neur.roiInfo[x][targ]['pre'] * (Neur.roiInfo[x][sour]['post'] / Neur.loc[x, 'post']) for x in range(len(Neur))]

                    new_tbars = [Neur.roiInfo[x][targ]['pre'] for x in range(len(Neur))]

                    # body_ids
                    body_ids.append(Neur.bodyId.values)

                    if Neur.roiInfo.shape[0] > 0:
                        summed_connectivity += np.sum(conn_strengths)
                        weighted_synapse_number += np.sum(weighted_synapses)
                        weak_neurons += n_weak
                        medium_neurons += n_medium
                        strong_neurons += n_strong
                        tbars += np.sum(new_tbars)

            WeakConnections.loc[[roi_source], [roi_target]] = weak_neurons
            MediumConnections.loc[[roi_source], [roi_target]] = medium_neurons
            StrongConnections.loc[[roi_source], [roi_target]] = strong_neurons

            Connectivity.loc[[roi_source], [roi_target]] = summed_connectivity
            WeightedSynapseNumber.loc[[roi_source], [roi_target]] = weighted_synapse_number
            TBars.loc[[roi_source], [roi_target]] = tbars

    body_ids = np.unique(np.hstack(body_ids)) # don't double count cells that contribute to multiple connections

    return WeakConnections, MediumConnections, StrongConnections, Connectivity, WeightedSynapseNumber, TBars, body_ids


def fetchRegionConnectivity(neuprint_client, mapping=None):
    """
    Query neuprint for a region-to-region synaptic weight matrix.

    Uses a single bulk fetch_adjacencies call instead of the O(N^2) fetch_neurons
    loop in computeConnectivityMatrix. Each neuron is assigned to the atlas ROI
    where it has the most output synapses (pre) for the source axis, and the most
    input synapses (post) for the target axis.

    Parameters:
    - neuprint_client: neuprint Client instance
    - mapping: dict mapping atlas ROI names -> list of neuprint ROI names.
               Defaults to bridge.getRoiMapping().

    Returns:
    - region_matrix: pd.DataFrame (n_rois x n_rois), total synapse weight per region pair
    - body_ids: np.array of body IDs included
    """
    if mapping is None:
        mapping = bridge.getRoiMapping()

    rois = sorted(mapping.keys())

    # Reverse map: neuprint ROI name -> atlas ROI name
    neuprint_to_atlas = {}
    for atlas_roi, neuprint_rois in mapping.items():
        for nr in neuprint_rois:
            neuprint_to_atlas[nr] = atlas_roi

    neuprint_rois_flat = list(neuprint_to_atlas.keys())

    # Fetch all traced, uncropped neurons present in our ROIs
    neurons_df, _ = fetch_neurons(NeuronCriteria(
        status='Traced', cropped=False,
        inputRois=neuprint_rois_flat,
        outputRois=neuprint_rois_flat,
        roi_req='any',
    ))
    body_ids = np.array(neurons_df['bodyId'].tolist())

    # Assign each neuron to its primary output ROI (pre) and primary input ROI (post)
    output_roi_lookup = {}  # bodyId -> atlas ROI where neuron sends most output
    input_roi_lookup = {}   # bodyId -> atlas ROI where neuron receives most input
    for _, row in neurons_df.iterrows():
        bid = row['bodyId']
        best_out, best_out_n = None, 0
        best_in, best_in_n = None, 0
        for nr, counts in row['roiInfo'].items():
            if nr not in neuprint_to_atlas:
                continue
            atlas_roi = neuprint_to_atlas[nr]
            if counts.get('pre', 0) > best_out_n:
                best_out, best_out_n = atlas_roi, counts['pre']
            if counts.get('post', 0) > best_in_n:
                best_in, best_in_n = atlas_roi, counts['post']
        output_roi_lookup[bid] = best_out
        input_roi_lookup[bid] = best_in

    # Single bulk adjacency fetch
    _, conn_df = fetch_adjacencies(body_ids.tolist(), body_ids.tolist(), min_total_weight=1)

    # Map neuron pairs to atlas ROI pairs and sum weights
    conn_df['atlas_pre'] = conn_df['bodyId_pre'].map(output_roi_lookup)
    conn_df['atlas_post'] = conn_df['bodyId_post'].map(input_roi_lookup)
    conn_df = conn_df.dropna(subset=['atlas_pre', 'atlas_post'])

    region_matrix = (conn_df
                     .groupby(['atlas_pre', 'atlas_post'])['weight']
                     .sum()
                     .unstack(fill_value=0)
                     .reindex(index=rois, columns=rois, fill_value=0))

    return region_matrix, body_ids


def fetchSynapticMatrix(neuprint_client, mapping=None, min_weight=1):
    """
    Query neuprint for the full neuron-to-neuron synaptic connectivity matrix.

    Parameters:
    - neuprint_client: neuprint Client instance
    - mapping: dict mapping atlas ROI names -> list of neuprint ROI names.
               Defaults to bridge.getRoiMapping().
    - min_weight: int, minimum synapse count to include a connection (maps to min_total_weight).

    Returns:
    - W: scipy.sparse.csr_matrix (N x N), W[i, j] = total synapse count from neuron i to j
    - body_ids: np.array of body IDs; body_ids[i] corresponds to row/col i in W
    - region_labels: np.array of atlas ROI names; region_labels[i] is the primary input region of neuron i
    """
    if mapping is None:
        mapping = bridge.getRoiMapping()

    neuprint_rois_flat = [nr for sublist in mapping.values() for nr in sublist]

    neuprint_to_atlas = {nr: atlas_roi
                         for atlas_roi, neuprint_rois in mapping.items()
                         for nr in neuprint_rois}

    neurons_df, _ = fetch_neurons(NeuronCriteria(
        status='Traced', cropped=False,
        inputRois=neuprint_rois_flat,
        roi_req='any',
    ))

    # Filter to neurons whose primary input ROI (most post-synaptic sites) maps
    # to one of the atlas regions — proxy for cell body location.
    def primary_input_roi(roi_info):
        best_roi, best_n = None, 0
        for nr, counts in roi_info.items():
            if nr in neuprint_to_atlas and counts.get('post', 0) > best_n:
                best_roi, best_n = neuprint_to_atlas[nr], counts['post']
        return best_roi

    neurons_df = neurons_df.copy()
    neurons_df['atlas_roi'] = neurons_df['roiInfo'].map(primary_input_roi)
    neurons_df = neurons_df[neurons_df['atlas_roi'].notna()]

    # Order neurons by region map order, then by body ID within each region
    roi_order = {roi: i for i, roi in enumerate(mapping.keys())}
    neurons_df = neurons_df.sort_values(
        by=['atlas_roi', 'bodyId'],
        key=lambda col: col.map(roi_order) if col.name == 'atlas_roi' else col
    ).reset_index(drop=True)

    body_ids = neurons_df['bodyId'].to_numpy()
    region_labels = neurons_df['atlas_roi'].to_numpy()
    id_index = {bid: i for i, bid in enumerate(body_ids)}

    # Bulk fetch; aggregate per-ROI rows into a single weight per neuron pair
    _, conn_df = fetch_adjacencies(body_ids.tolist(), body_ids.tolist(), min_total_weight=min_weight)
    conn_agg = (conn_df
                .groupby(['bodyId_pre', 'bodyId_post'])['weight']
                .sum()
                .reset_index())

    rows = conn_agg['bodyId_pre'].map(id_index).values
    cols = conn_agg['bodyId_post'].map(id_index).values
    N = len(body_ids)
    W = csr_matrix((conn_agg['weight'].values, (rows, cols)), shape=(N, N))

    return W, body_ids, region_labels
