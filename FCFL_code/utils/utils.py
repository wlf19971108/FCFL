def exp_details(args):
    print('\nExperimental details:')
    print(f'    Method     : {args.method}')
    if args.method == "qfedavg":
        print(f'q value: {args.q}')    
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
        # if args.unequal:
        #     print('Unequal')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.bs_train}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def worst_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=False)
    worst = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(worst)/len(worst)

def best_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=True)
    best = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(best)/len(best)