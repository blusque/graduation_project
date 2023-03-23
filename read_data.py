m, n, k = [0, 0, 0]
with open('./senbai/data666.txt', 'r') as fobj:
    m, n, k = fobj.readline().strip().split(' ')[2 : 5]
    print(m, n, k)
    # for line in fobj.readlines():
    #     # print(line)
    #     if line == '\n':
    #         count += 1
    #         print(columns)
    #         columns = 0
    #     else:
    #         columns += 1
    #     data = line.split('\t')
    #     # print(len(data))
            
    # print(count)