
fold_set = set()
lines = open('fashionE_list.txt').readlines()
for line in lines:
    fold, filename = line.strip().split('/')
    fold_set.add(fold)


seed = int(len(fold_set) * 0.8)
print(seed)
fold_list = list(fold_set)
train_fold_list = fold_list[:seed]
test_fold_list = fold_list[seed:]

print(len(fold_list))
print(len(train_fold_list))
print(len(test_fold_list))


file = open('fashionE_train_test.txt', 'a')
lines = open('fashionE_list.txt').readlines()
for line in lines:
    fold, filename = line.strip().split('/')
    if train_fold_list.count(fold) > 0:
        line = line.strip() + '\t' + 'train' + '\n'
    elif test_fold_list.count(fold) > 0:
        line = line.strip() + '\t' + 'test' + '\n'
    file.write(line)
file.close()

print("END")
