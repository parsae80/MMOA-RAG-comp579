# import requests

# def query_search_api(N=10):
#     url = 'http://10.215.192.149:8000/search'
#     params = {'N': N}
#     response = requests.get(url, params=params)
#     return response.json()

# if __name__ == '__main__':
#     results = query_search_api(15)
#     print(results)

# import requests
# import time

# url = 'http://10.215.192.149:8000/search'
# data = {
#     'questions': ['What is AI?', 'How does machine learning work?'],
#     'N': 10
# }

# start_time = time.time()
# response = requests.post(url, json=data)
# end_time = time.time()
# print('time consuming: {} seconds'.format(end_time - start_time))

# print(response)


import requests
import time

def extract_questions_and_docs(response):
    """
    从HTTP响应中提取问题和对应的文档列表。

    参数：
    - response: requests.Response 对象，HTTP请求的响应。

    返回：
    - 提取的信息列表，每个元素是一个字典，包含 'question' 和 'top_k_docs'。
    """
    # # 检查响应状态码
    # if response.status_code != 200:
    #     print(f"Request failed with status code: {response.status_code}")
    #     return []

    # 解析 JSON 响应
    results = response.json()

    # 提取信息
    extracted_info_dict = {}
    for result in results:
        question = result.get('question')
        top_k_docs = result.get('top_k_docs', [])
        
        # # 输出或处理结果
        # print(f"Question: {question}")
        # print("Top K Documents:")
        # for doc in top_k_docs:
        #     print(f"- {doc}")
        # print()  # 添加空行以便分隔不同问题的结果
        
        # 添加到列表中
        extracted_info_dict[question] = top_k_docs
        # extracted_info.append({'question': question, 'top_k_docs': top_k_docs})

    return extracted_info_dict

# 示例调用
url = 'http://10.215.192.149:8000/search'
questions = [
        'What is AI?', 
        'How does machine learning work?', 
        'which city is the capital of China?', 
        'What are the implications of artificial intelligence on the concept of free will, and how might this affect our understanding of moral responsibility?',
        'How do climate change and biodiversity loss interact, and what are the most effective strategies for mitigating these interconnected global challenges?',
        'Explain the concept of a zero-day exploit in cybersecurity and discuss the balance between the need for transparency in vulnerability disclosure and the potential risks to national security.',
        'the director of iron man.'
        ]
questions = questions + questions
questions = questions + questions
# questions = questions + questions
print(len(questions))
data = {
    'questions': questions,
    'N': 10
}

start_time = time.time()
response = requests.post(url, json=data)
end_time = time.time()
print('time consuming: {} seconds'.format(end_time - start_time))

# 提取并输出信息
extracted_info_dict = extract_questions_and_docs(response)