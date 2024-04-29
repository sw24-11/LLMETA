import time

f=open('C:/Users/kbh/Code/expr.txt', 'w')

for path in model_paths:
    for n_batch in n_batches:
        llm = llama2_chain(path, n_batch=n_batch, n_gpu_layers=n_gpu)
        llm_chain, prompt = llm.llm_set()
        cnt=0
        avg_time = 0
        for i in range(100):
            start_time = time.time()
            response = llm_chain.invoke(prompt)
            end_time = time.time()
            if response['text'].find('Title') != -1 and response['text'].find('Authors') and response['text'].find('Abstract'):
                cnt+=1
            if i!=0:
                curr_time = end_time-start_time
                avg_time += curr_time
        f.write('===============================\n')
        f.write('model : %s\n' %path)
        f.write('batch : %d\n' %n_batch)
        f.write('acc : %d%%\n' %cnt)
        f.write('time : %.2f\n' %avg_time)
        f.write('===============================\n\n')
        f.flush()