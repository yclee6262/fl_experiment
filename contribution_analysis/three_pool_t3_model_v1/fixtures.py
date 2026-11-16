from fractions import Fraction as F


def poly(slope=0,quad=0,offset=1000):
    return dict(offset=str(F(offset)),linear=[str(F(slope)),'0'],quadratic=[[str(F(quad)),'0'],['0','0']])


def cases():
    out=[]
    def add(cat,i,models,weights,reference,Csign,Dsign):
        out.append(dict(id=cat+'_'+str(i),category=cat,split='development' if i<3 else 'heldout',models=models,
            weights=[str(F(v)) for v in weights],reference=reference,target=[0],anchor_weights=['1/2','1/2'],
            anchors=[['1','0'],['-1/4','0']],start=['0','0'],expected_C_sign=Csign,expected_D_sign=Dsign,eta='1/10'))
    halves=[F(1,2),F(1,2)];bs=[1,F(6,5),F(7,5),F(11,10),F(13,10),F(3,2)]
    for i,b in enumerate([1,2,3,F(3,2),F(5,2),F(7,2)]):add('agreement',i,[poly(b),poly(b)],halves,poly(b),0,0)
    for i,b in enumerate(bs):add('useful_information',i,[poly(1),poly(-b)],[F(3,4),F(1,4)],poly(1),1,1)
    for i,b in enumerate([2,3,4,F(5,2),F(7,2),F(9,2)]):add('accurate_redundant',i,[poly(1),poly(b)],halves,poly(1),0,0)
    for i,q in enumerate([20,22,24,21,23,25]):add('local_error',i,[poly(1,-2*q),poly(1)],halves,poly(1),-1,-1)
    for i,b in enumerate([50,100,150,75,125,175]):add('fixed_bias',i,[poly(1,offset=1000+b),poly(1)],halves,poly(1),0,0)
    for i,scale in enumerate([2,4,8,3,6,10]):add('individual_scale',i,[poly(scale,offset=1000*scale),poly(-1)],halves,poly(-1),1,-1)
    for i,b in enumerate(bs):add('optimistic_acceptance',i,[poly(-b),poly(1)],[F(3,4),F(1,4)],poly(1),1,-1)
    for i,q in enumerate([1,2,3,F(3,2),F(5,2),F(7,2)]):add('prevent_bad_step',i,[poly(1,2*q),poly(-1)],halves,poly(0,q),1,1)
    for i,q in enumerate([1,2,3,F(3,2),F(5,2),F(7,2)]):add('both_no_update',i,[poly(0,q),poly(0,2*q)],halves,poly(0,q),0,0)
    for i,eps in enumerate([F(1,10),F(1,100),F(0),F(1,5),F(1,1000),F(1)]):
        add('concentrated_weights',i,[poly(1),poly(-1)],[1-eps,eps],poly(1),None if eps==0 else 0 if eps==1 else 1,None if eps==0 else 0 if eps==1 else 1)
    return out
