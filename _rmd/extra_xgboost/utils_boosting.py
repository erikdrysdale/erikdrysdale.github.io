##############################################################
###### -------- UTILITY FUNCTIONS FOR PYTHON --------- #######


def head(x,k=6,tail=False):
    if len(x.shape)==1:
        # Check dimensions
        if x.shape[0]<k:
            k1 = x.shape[0]
        else:
            k1 = k
        # Print head/tail
        if tail:
            print(x[-k1:])
        else:
            print(x[0:k1])
    elif len(x.shape)==2:
        # Check dimensions
        if x.shape[0]<k:
            k1 = x.shape[0]
        else:
            k1 = k
        if x.shape[1]<k:
            k2 = x.shape[1]
        else:
            k2 = k
        # Print head/tail
        if tail:
            print(x[-k1:,-k2:])
        else:
            print(x[0:k1,0:k2])
    else:
        print('Nothing')