import tensorflow as tf
import pickle as pkl
import numpy as np
import functools
import model
import estimator
import random

class generate_data(object):
    def __init__(self,cat_voc_file,mid_voc_file,uid_voc_file, train_file, test_file,batch_size=32):
        f = open(cat_voc_file,'rb')
        f1 = open(mid_voc_file,'rb')
        f2 = open(uid_voc_file,'rb')
        self.cat_dict = pkl.load(f)
        self.mid_dict = pkl.load(f1)
        self.uid_dict = pkl.load(f2)
        f3 = open(train_file,'r')
        f4 = open(test_file,'r')
        self.train = f3.readlines()
        self.test = f4.readlines()
        self.batch_size = batch_size
        self.n_uid = len(self.uid_dict) + 1
        self.n_mid = len(self.mid_dict) + 1
        self.n_cat = len(self.cat_dict) + 1
        f.close()
        f1.close()
        f3.close()
        f4.close()
             
    def batch_fn(self,batch_data):
        tmp_items = []
        tmp_cats = []
        for prev_items in batch_data['mid']:
            tmp_items.append(np.array(prev_items,dtype=np.int64))
        for prev_cat in batch_data['mid_cat']:
            tmp_cats.append(np.array(prev_cat,dtype=np.int64))
        lens = np.array(batch_data['mid_len'],dtype=np.int64)
        mask1 = lens[:,None] > np.arange(lens.max())
        mask2 = lens[:,None] > np.arange(lens.max())
        result_mid = np.empty(mask1.shape,dtype=np.int64)
        result_cat = np.empty(mask2.shape,dtype=np.int64)
        result_mid.fill(0)
        result_cat.fill(0)
        result_mid[mask1] = np.concatenate(batch_data['mid'])
        result_cat[mask2] = np.concatenate(batch_data['mid_cat'])
        batch_data['mid'] = result_mid
        batch_data['mid_cat'] = result_cat
        batch_data['label'] = np.array(batch_data['label'],dtype=np.int64)
        batch_data['uid'] = np.array(batch_data['uid'],dtype=np.int64)
        batch_data['target_mid'] = np.array(batch_data['target_mid'],dtype=np.int64)
        batch_data['target_cat'] = np.array(batch_data['target_cat'],dtype=np.int64)
        batch_data['mask'] = np.where(mask1,1,0)
        #print(batch_data)
        return batch_data

    def build_batch_generator(self):
        trainset = {}
        testset = {}
        label = []
        uid = []
        mid = []
        mid_len = []
        mid_cat = []
        target_mid = []
        target_cat = []
        batch_data = {}
        i = 1
        random.shuffle(self.train)
        random.shuffle(self.test)
        for line in self.train:
            line = line.strip().split('\t')
            label.append(int(line[0]))
            uid.append(self.uid_dict[line[1]] if line[1] in self.uid_dict else 0)
            target_mid.append(self.mid_dict[line[2]] if line[2] in self.mid_dict else 0)
            target_cat.append(self.cat_dict[line[3]] if line[3] in self.cat_dict else 0)
            hist = line[4].split('')
            prev_items = [self.mid_dict[item] if item in self.mid_dict else 0 for item in hist]
            mid_len.append(len(prev_items))
            mid.append(prev_items)
            hist_cat = line[5].split('')
            prev_cats = [self.cat_dict[cat] if cat in self.cat_dict else 0 for cat in hist_cat]
            mid_cat.append(prev_cats)
            if i % (self.batch_size) == 0:
                batch_data['label'] = label
                batch_data['uid'] = uid
                batch_data['target_mid'] = target_mid
                batch_data['target_cat'] = target_cat
                batch_data['mid'] = mid
                batch_data['mid_cat'] = mid_cat
                batch_data['mid_len'] = mid_len
                yield self.batch_fn(batch_data)
                label = []
                uid = []
                target_mid = []
                target_cat = []
                mid = []
                mid_cat = []
                mid_len = []
                batch_data = {}
            i += 1

    def build_dataset_generator(self):
        batch_generator = self.build_batch_generator()
        for batch in batch_generator:
            yield batch

    def input_fn_with_generator(self,generator,batch_size=32,buffer_size=32):
        output_types = {}
        output_types.update({'label':tf.int64})
        output_types.update({'mid':tf.int64})
        output_types.update({'uid':tf.int64})
        output_types.update({'mid_cat':tf.int64})
        output_types.update({'mid_len':tf.int64})
        output_types.update({'target_mid':tf.int64})
        output_types.update({'target_cat':tf.int64})
        output_types.update({'mask':tf.int64})
        output_shapes = {}
        output_shapes.update({'label':[batch_size]})
        output_shapes.update({'mid':[batch_size,None]})
        output_shapes.update({'mid_cat':[batch_size,None]})
        output_shapes.update({'mid_len':[batch_size]})
        output_shapes.update({'target_mid':[batch_size]})
        output_shapes.update({'target_cat':[batch_size]})
        output_shapes.update({'uid':[batch_size]})
        output_shapes.update({'mask':[batch_size,None]})
        dataset = tf.data.Dataset.from_generator(lambda:generator,output_types=output_types,output_shapes=output_shapes)
        dataset = dataset.map(lambda batch:(batch,batch.pop('label')))
        dataset = dataset.prefetch(buffer_size=None)
        return dataset

if __name__ == '__main__':
    mode = 'train'
    data = generate_data('cat_voc.pkl','mid_voc.pkl','uid_voc.pkl','local_train_splitByUser','local_test_splitByUser',32)
    dataset_generator = data.build_dataset_generator()
    n_mid,n_uid,n_cat = data.n_mid,data.n_uid,data.n_cat
    input_fn = functools.partial(data.input_fn_with_generator, generator=dataset_generator)
    estimator_run_config = tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=1000,
            log_step_count_steps=100)
    model_fn = estimator.model_fn
    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir='./model',
            config=estimator_run_config,
            params={
                'batch_size':32,
                'hidden_size':32,
                'embedding_size':64,
                'attention_size':64,
                'n_mid':n_mid,
                'n_uid':n_uid,
                'n_cat':n_cat
                }
            )
    if mode == 'train':
        print('train begin!')
        train_spec = tf.estimator.TrainSpec(input_fn=input_fn,max_steps=200000)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn,steps=1000)
        tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
        print('train_end!')
    



        

    
