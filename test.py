import numpy as np
import random

q_table = np.zeros([3, 3]) # Q-table 초기화
lr = 0.1 # 학습률
gamma = 0.99 # 감마값이란 
epsilon = 0.01 # 탐욕 정책용 엡실론 값 #앱실론 값이 낮을 수록 탐험을 많이 하고, 높을 수록 탐험을 적게 한다. 
doors = [0, 1, 2] 

def choose_action(state,opendoor):# 행동 선택
    if np.random.uniform() < epsilon:# 탐욕 정책
        action = np.random.choice(list(set(doors)-set([state,opendoor])))
    else:
        action = np.argmax(q_table[state])# Q-table에서 최대값 행동
    return action

def montihol(time):
    suc = 0 # 자동차를 고른 횟수
    fail = 0 # 염소를 고른 횟수
    
    for i in range(time):
        reward = random.choice(doors) # 자동차 위치
        state = random.choice(doors) # 문 선택
        opendoor=list(set(doors)-set([reward,state]))[0] # 열린 문(염소)
        action = choose_action(state,opendoor)
        
        if action == reward:
            fail += 1
            r = -1 # 실패 시 -1의 보상
        else:
            suc += 1
            r = 1 # 성공 시 +1의 보상

        # Q-table 업데이트
        q_table[state, action] = (1-lr) * q_table[state, action] + lr * (r + gamma * np.max(q_table[reward, :]))
        
    arr = [suc,fail,suc+fail,suc/(suc+fail)] # 리턴할때 성공, 실패, 총 경우, 성공확률 리스트 리턴
    return arr
tmp=[]
for i in range(10):
    arr = montihol(100)
    print('성공 %d번, 실패 %d번, 성공확률 %f%%입니다.'%(arr[0],arr[1],arr[3]*100))
    print(q_table)
    # arr초기화
    tmp.append(arr)
    arr = [0,0,0,0]
print("성공횟수 평균 : %f"%(sum([i[0] for i in tmp])/len(tmp)))



    
