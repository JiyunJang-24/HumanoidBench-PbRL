from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp
from collections import defaultdict

import optax
import numpy as np
from flax.training.train_state import TrainState
from tqdm import tqdm, trange
from flax.training.early_stopping import EarlyStopping
#from viskit.logging import logger, setup_logger

import os
from PIL import Image
from matplotlib import pyplot as plt
from .jax_utils import batch_to_jax, init_rng, next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss
from .utils import Timer, set_random_seed, prefix_metrics, save_pickle


# class PrefTransformer_Trainer():
    
#     def __init__(self, reward_model):
#         self.reward_model = reward_model
        
#     def update(obs, action, reward, episode, task, data_pair, save_dir, env_name, num_query, query_len, save_label):
#         metrics, epoch = self.reward_model.train_RLHF(obs, action, reward, episode, task, data_pair, save_dir, env_name, num_query, query_len, save_label)
#         if self.reward_model.config.save_model:
#             return metrics, epoch, self.reward_model
#         else:
#             return metrics, epoch, None
        
        
    

class PrefTransformer(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 3
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 1
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.train_type = "mean"

        # Weighted Sum option
        config.use_weighted_sum = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            'CosineDecay': optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(
                        value=self.config.trans_lr
                    )
                ],
                [self.config.warmup_steps]
            ),
            'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)
        seed = 42
        init_rng(seed)
        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32)
        )
        self._train_states['trans'] = TrainState.create(
            params=trans_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['trans']
        self._model_keys = tuple(model_keys)
        self._total_steps = 0
    
    def index_batch(self, batch, indices):
        indexed = {}
        for key in batch.keys():
            indexed[key] = batch[key][indices, ...]
        return indexed
    
    def train_RLHF(self, obs, action, reward, episode, task, data_pair, save_dir, env_name, num_query, query_len, save_label, logger, logger_info):
        true_eval = True 
        label_type = 1
        #label_type = 0 # perfectly rational

        pref_dataset, pref_eval_dataset = self.load_queries_with_indices(
            obs=obs, action=action, reward=reward, episode=episode, task=task, data_pair=data_pair,
            num_query=num_query, len_query=query_len,
            label_type=label_type, saved_labels=save_label,
            balance=False, scripted_teacher=False)
        
        seed = 42
        set_random_seed(seed)
        observation_dim = self.observation_dim
        action_dim = self.action_dim

        data_size = pref_dataset["observations"].shape[0]
        
        interval = data_size / self.config.batch_size_RLHF

        if (interval - int(interval)) > 0:
            interval = int(interval) + 1
        interval = int(interval)
        
        
        eval_data_size = pref_eval_dataset["observations"].shape[0]
        eval_interval = eval_data_size / self.config.batch_size_RLHF_eval
        
        if (eval_interval - int(eval_interval)) > 0:
            eval_interval = int(eval_interval) + 1
        
        eval_interval = int(eval_interval)
        
        print('eval_data_size: ', eval_data_size, ' eval_interval : ', eval_interval)
        early_stop = EarlyStopping(min_delta=self.config.min_delta, patience=self.config.patience)
        train_loss = "reward/trans_loss"
        
        batch_size = self.config.batch_size_RLHF
        batch_size_eval = self.config.batch_size_RLHF_eval
        criteria_key = None
        ## DW
        total_metrics = {'cse_loss' : [], 'trans_loss' : [], 'eval_cse_loss' : [], 'eval_trans_loss' : []}
        
        for epoch in range(self.config.n_epochs + 1):
            metrics = defaultdict(list)
            metrics['epoch'] = epoch
            if epoch:
                # train phase
                shuffled_idx = np.random.permutation(pref_dataset["observations"].shape[0])
                for i in range(interval):
                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, pref_dataset["observations"].shape[0])
                    with Timer() as train_timer:
                        # train
                        batch = batch_to_jax(self.index_batch(pref_dataset, shuffled_idx[start_pt:end_pt]))
                        for key, val in prefix_metrics(self.train(batch), 'reward').items():
                            metrics[key].append(val)
                metrics['train_time'] = train_timer()
            else:
                # for using early stopping with train loss.
                metrics[train_loss] = [float(query_len)]

            # eval phase
            if epoch % self.config.eval_period == 0:
                for j in range(eval_interval):
                    eval_start_pt, eval_end_pt = j * batch_size_eval, min((j + 1) * batch_size_eval, pref_eval_dataset["observations"].shape[0])
                    # batch_eval = batch_to_jax(index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                    batch_eval = batch_to_jax(self.index_batch(pref_eval_dataset, range(eval_start_pt, eval_end_pt)))
                    for key, val in prefix_metrics(self.evaluation(batch_eval), 'reward').items():
                        metrics[key].append(val)
                if not criteria_key:
                    criteria_key = train_loss
                    # if "humanoid" in env_name and not "dense" in env_name: #and not true_eval:
                    #     # choose train loss as criteria.
                    #     criteria_key = train_loss
                    # else:
                    #     # choose eval loss as criteria.
                    #     criteria_key = key
                criteria = np.mean(metrics[criteria_key])
                #print(early_stop.update(criteria))
                early_stop.update(criteria)
                #has_improved, early_stop = early_stop.update(criteria)
                if early_stop.should_stop and self.config.early_stop:
                    for key, val in metrics.items():
                        if isinstance(val, list):
                            metrics[key] = np.mean(val)
                    #logger.record_dict(metrics)
                    #logger.dump_tabular(with_prefix=False, with_timestamp=False)
                    #wb_logger.log(metrics)
                    print('Met early stopping criteria, breaking...')
                    break
                elif epoch > 0 and early_stop.has_improved:
                    metrics["best_epoch"] = epoch
                    metrics[f"{key}_best"] = criteria
                    save_data = {"reward_model": self, "epoch": epoch}
                    save_pickle(save_data, "best_model.pkl", save_dir+'/'+env_name)

            for key, val in metrics.items():
                if isinstance(val, list):
                    metrics[key] = np.mean(val)
            #logger.record_dict(metrics)
            #logger.dump_tabular(with_prefix=False, with_timestamp=False)
            #wb_logger.log(metrics)
            
            ## DW
            for k, v in metrics.items():
                k_base = os.path.basename(k)
                if k_base in total_metrics.keys():
                    total_metrics[k_base].append([v, epoch])
         
        if self.config.save_model:
            save_data = {'reward_model': self, 'epoch': epoch}
            save_pickle(save_data, 'model.pkl', save_dir+'/'+env_name)
        
        ## DW
        final_metric = defaultdict(list)
        for k, _ in total_metrics.items():
            fig, ax = plt.subplots()
            values = total_metrics[k]
            x_values = []
            y_values = []
            for value in values:
                x_values.append(value[1])
                y_values.append(value[0])
            
            ax.plot(x_values, y_values, color="black")
            ax.set_title(k)
            img = self.fig2img(fig)
            final_metric[k] = img
        
        final_metric['step'] = logger_info['global_step']
        logger.log(final_metric, 'reward_model')
                
        # return metrics
    
    def load_queries_with_indices(self, obs, action, reward, episode, task, data_pair, num_query, len_query, label_type, saved_labels, balance=False, scripted_teacher=False):
        # 데이터셋 로드    
        total_reward_seq_1, total_reward_seq_2 = np.zeros((num_query, len_query)), np.zeros((num_query, len_query))
        observation_dim = obs.shape[-1]
        action_dim = action.shape[-1]

        total_obs_seq_1, total_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
        total_next_obs_seq_1, total_next_obs_seq_2 = np.zeros((num_query, len_query, observation_dim)), np.zeros((num_query, len_query, observation_dim))
        total_act_seq_1, total_act_seq_2 = np.zeros((num_query, len_query, action_dim)), np.zeros((num_query, len_query, action_dim))
        total_timestep_1, total_timestep_2 = np.zeros((num_query, len_query), dtype=np.int32), np.zeros((num_query, len_query), dtype=np.int32)


        for seg_idx in trange(num_query):

            start_1 = data_pair[seg_idx][0]
            start_2 = data_pair[seg_idx][1]

            for t in trange(len_query-1, leave=False):
                state_1 = obs[t][start_1].cpu().numpy()  
                action_1 = action[t][start_1].cpu().numpy()
                reward_1 = reward[t][start_1].cpu().numpy()
                next_observation_1 = obs[t+1][start_1].cpu().numpy()
                
                total_reward_seq_1[seg_idx, t] = reward_1
                total_obs_seq_1[seg_idx, t, :] = state_1
                total_next_obs_seq_1[seg_idx, t, :] = next_observation_1
                total_act_seq_1[seg_idx, t, :] = action_1
                total_timestep_1[seg_idx, t] = t

            for t in trange(len_query-1, leave=False):
                state_2 = obs[t][start_2].cpu().numpy()  
                action_2 = action[t][start_2].cpu().numpy()
                reward_2 = reward[t][start_2].cpu().numpy()
                next_observation_2 = obs[t+1][start_2].cpu().numpy()
                
                total_reward_seq_2[seg_idx, t] = reward_2
                total_obs_seq_2[seg_idx, t, :] = state_2
                total_next_obs_seq_2[seg_idx, t, :] = next_observation_2
                total_act_seq_2[seg_idx, t, :] = action_2
                total_timestep_2[seg_idx, t] = t

        # if saved_labels is None:
        #     query_range = np.arange(num_query)
        # else:
        #     query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))

        # for query_count, i in enumerate(tqdm(query_range, desc="get queries from saved indices")):
        #     temp_count = 0
        #     while(temp_count < 2):                
        #         start_idx = saved_indices[temp_count][i]
        #         end_idx = start_idx + len_query

        #         reward_seq = rewards[start_idx:end_idx]
        #         obs_seq = observations[start_idx:end_idx]
        #         next_obs_seq = next_observations[start_idx:end_idx]
        #         act_seq = actions[start_idx:end_idx]
        #         timestep_seq = np.arange(1, len_query + 1)

        #         if temp_count == 0:
        #             total_reward_seq_1[query_count] = reward_seq
        #             total_obs_seq_1[query_count] = obs_seq
        #             total_next_obs_seq_1[query_count] = next_obs_seq
        #             total_act_seq_1[query_count] = act_seq
        #             total_timestep_1[query_count] = timestep_seq
        #         else:
        #             total_reward_seq_2[query_count] = reward_seq
        #             total_obs_seq_2[query_count] = obs_seq
        #             total_next_obs_seq_2[query_count] = next_obs_seq
        #             total_act_seq_2[query_count] = act_seq
        #             total_timestep_2[query_count] = timestep_seq
                        
        #         temp_count += 1
                
        seg_reward_1 = total_reward_seq_1.copy()
        seg_reward_2 = total_reward_seq_2.copy()
        
        seg_obs_1 = total_obs_seq_1.copy()
        seg_obs_2 = total_obs_seq_2.copy()
        
        seg_next_obs_1 = total_next_obs_seq_1.copy()
        seg_next_obs_2 = total_next_obs_seq_2.copy()
        
        seq_act_1 = total_act_seq_1.copy()
        seq_act_2 = total_act_seq_2.copy()

        seq_timestep_1 = total_timestep_1.copy()
        seq_timestep_2 = total_timestep_2.copy()
    
        if label_type == 0: # perfectly rational
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        elif label_type == 1:
            sum_r_t_1 = np.sum(seg_reward_1, axis=1)
            sum_r_t_2 = np.sum(seg_reward_2, axis=1)
            binary_label = 1*(sum_r_t_1 < sum_r_t_2)
            rational_labels = np.zeros((len(binary_label), 2))
            rational_labels[np.arange(binary_label.size), binary_label] = 1.0
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= 0).reshape(-1)
            rational_labels[margin_index] = 0.5

        batch = {}
        if scripted_teacher:
            batch['labels'] = rational_labels
        else:
            # human_labels = np.zeros((len(saved_labels), 2))
            # human_labels[np.array(saved_labels)==0,0] = 1.
            # human_labels[np.array(saved_labels)==1,1] = 1.
            # human_labels[np.array(saved_labels)==-1] = 0.5
            #human_labels = human_labels[query_range]
            batch['labels'] = saved_labels
        batch['script_labels'] = rational_labels

        batch['observations'] = seg_obs_1 # for compatibility, remove "_1"
        batch['next_observations'] = seg_next_obs_1
        batch['actions'] = seq_act_1
        batch['observations_2'] = seg_obs_2
        batch['next_observations_2'] = seg_next_obs_2
        batch['actions_2'] = seq_act_2
        batch['timestep_1'] = seq_timestep_1
        batch['timestep_2'] = seq_timestep_2
        #batch['start_indices'] = saved_indices[0]
        #batch['start_indices_2'] = saved_indices[1]
        
        if balance:
            nonzero_condition = np.any(batch["labels"] != [0.5, 0.5], axis=1)
            nonzero_idx, = np.where(nonzero_condition)
            zero_idx, = np.where(np.logical_not(nonzero_condition))
            selected_zero_idx = np.random.choice(zero_idx, len(nonzero_idx))
            for key, val in batch.items():
                batch[key] = val[np.concatenate([selected_zero_idx, nonzero_idx])]
            print(f"size of batch after balancing: {len(batch['labels'])}")

        total_samples = len(batch['observations'])
        split_index = int(total_samples * 0.75)

        # Create batch_train and batch_eval
        batch_train = {key: value[:split_index] for key, value in batch.items()}
        batch_eval = {key: value[split_index:] for key, value in batch.items()}
        
        
        return batch_train, batch_eval
    
    
    def evaluation(self, batch):
        metrics = self._eval_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    def get_reward(self, batch):
        return self._get_reward_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=('self'))
    def _get_reward_step(self, train_states, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        # n_obs = batch['next_observations']
        attn_mask = batch['attn_mask']

        train_params = {key: train_states[key].params for key in self.model_keys}
        trans_pred, attn_weights = self.trans.apply(train_params['trans'], obs, act, timestep, attn_mask=attn_mask, reverse=False)
        return trans_pred["value"], attn_weights[-1]
  
    @partial(jax.jit, static_argnames=('self'))
    def _eval_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=False, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=False, attn_mask=None, rngs={"dropout": rng})
            
            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
          
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
         
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
          
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss
            loss_collection['trans'] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_cse_loss=aux_values['cse_loss'],
            eval_trans_loss=aux_values['trans_loss'],
        )

        return metrics
      
    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch
        )
        return metrics

    @partial(jax.jit, static_argnames=('self'))
    def _train_pref_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
          
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
           
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
           
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
           
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)
            cse_loss = trans_loss

            loss_collection['trans'] = trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            cse_loss=aux_values['cse_loss'],
            trans_loss=aux_values['trans_loss'],
        )

        return new_train_states, metrics

    def train_semi(self, labeled_batch, unlabeled_batch, lmd, tau):
        self._total_steps += 1
        self._train_states, metrics = self._train_semi_pref_step(
            self._train_states, labeled_batch, unlabeled_batch, lmd, tau, next_rng()
        )
        return metrics

    @partial(jax.jit, static_argnames=('self'))
    def _train_semi_pref_step(self, train_states, labeled_batch, unlabeled_batch, lmd, tau, rng):
        def compute_logits(train_params, batch, rng):
            obs_1 = batch['observations']
            act_1 = batch['actions']
            obs_2 = batch['observations_2']
            act_2 = batch['actions_2']
            timestep_1 = batch['timestep_1']
            timestep_2 = batch['timestep_2']
            labels = batch['labels']
         
            B, T, _ = batch['observations'].shape
            B, T, _ = batch['actions'].shape

            rng, _ = jax.random.split(rng)
           
            trans_pred_1, _ = self.trans.apply(train_params['trans'], obs_1, act_1, timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            trans_pred_2, _ = self.trans.apply(train_params['trans'], obs_2, act_2, timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            if self.config.use_weighted_sum:
                trans_pred_1 = trans_pred_1["weighted_sum"]
                trans_pred_2 = trans_pred_2["weighted_sum"]
            else:
                trans_pred_1 = trans_pred_1["value"]
                trans_pred_2 = trans_pred_2["value"]

            if self.config.train_type == "mean":
                sum_pred_1 = jnp.mean(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.mean(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "sum":
                sum_pred_1 = jnp.sum(trans_pred_1.reshape(B, T), axis=1).reshape(-1, 1)
                sum_pred_2 = jnp.sum(trans_pred_2.reshape(B, T), axis=1).reshape(-1, 1)
            elif self.config.train_type == "last":
                sum_pred_1 = trans_pred_1.reshape(B, T)[:, -1].reshape(-1, 1)
                sum_pred_2 = trans_pred_2.reshape(B, T)[:, -1].reshape(-1, 1)
           
            logits = jnp.concatenate([sum_pred_1, sum_pred_2], axis=1)
            return logits, labels

        def loss_fn(train_params, lmd, tau, rng):
            rng, _ = jax.random.split(rng)
            logits, labels = compute_logits(train_params, labeled_batch, rng)
            u_logits, _ = compute_logits(train_params, unlabeled_batch, rng)
                        
            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            
            """ reward function loss """
            label_target = jax.lax.stop_gradient(labels)
            trans_loss = cross_ent_loss(logits, label_target)

            u_confidence = jnp.max(jax.nn.softmax(u_logits, axis=-1), axis=-1)
            pseudo_labels = jnp.argmax(u_logits, axis=-1)
            pseudo_label_target = jax.lax.stop_gradient(pseudo_labels)
                    
            loss_ = optax.softmax_cross_entropy(logits=u_logits, labels=jax.nn.one_hot(pseudo_label_target, num_classes=2))
            u_trans_loss = jnp.sum(jnp.where(u_confidence > tau, loss_, 0)) / (jnp.count_nonzero(u_confidence > tau) + 1e-4)
            u_trans_ratio = jnp.count_nonzero(u_confidence > tau) / len(u_confidence) * 100

            # labeling neutral cases.
            binarized_idx = jnp.where(unlabeled_batch["labels"][:, 0] != 0.5, 1., 0.)
            real_label = jnp.argmax(unlabeled_batch["labels"], axis=-1)
            u_trans_acc = jnp.sum(jnp.where(pseudo_label_target == real_label, 1., 0.) * binarized_idx) / jnp.sum(binarized_idx) * 100

            loss_collection['trans'] = last_loss = trans_loss + lmd * u_trans_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, lmd, tau, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            trans_loss=aux_values['trans_loss'],
            u_trans_loss=aux_values['u_trans_loss'],
            last_loss=aux_values['last_loss'],
            u_trans_ratio=aux_values['u_trans_ratio'],
            u_train_acc=aux_values['u_trans_acc']
        )

        return new_train_states, metrics
   
    def train_regression(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_regression_step(
            self._train_states, next_rng(), batch
        )
        return metrics
   
    @partial(jax.jit, static_argnames=('self'))
    def _train_regression_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            next_observations = batch['next_observations']
            actions = batch['actions']
            rewards = batch['rewards']
           
            in_obs = jnp.concatenate([observations, next_observations], axis=-1)

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
           
            """ reward function loss """
            rf_pred = self.rf.apply(train_params['rf'], observations, actions)
            reward_target = jax.lax.stop_gradient(rewards)
            rf_loss = mse_loss(rf_pred, reward_target)

            loss_collection['rf'] = rf_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            rf_loss=aux_values['rf_loss'],
            average_rf=aux_values['rf_pred'].mean(),
        )

        return new_train_states, metrics
    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
    
    ## DW
    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
