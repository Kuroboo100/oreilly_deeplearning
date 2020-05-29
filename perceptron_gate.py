def and_gate(x1,x2):
    w1=0.3
    w2=0.3
    temp=x1*w1+x2*w2
    if temp>0.5:
        return 1
    else:
        return 0

def or_gate(x1,x2):
    w1=0.3
    w2=0.3
    temp=x1*w1+x2*w2
    if temp>0.2:
        return 1
    else:
        return 0

def nand_gate(x1,x2):
    w1=0.3
    w2=0.3
    temp=x1*w1+x2*w2
    if temp>0.5:
        return 0
    else:
        return 1

def xor_gate(x1,x2):
    return

def main():
    assert and_gate(1,1)==1
    assert and_gate(1,0)==0
    assert and_gate(0,1)==0
    assert and_gate(0,0)==0

    assert or_gate(1,1)==1
    assert or_gate(1,0)==1
    assert or_gate(0,1)==1
    assert or_gate(0,0)==0

    assert nand_gate(1,1)==0
    assert nand_gate(1,0)==1
    assert nand_gate(0,1)==1
    assert nand_gate(0,0)==1

    return

if __name__=="__main__":
    main()