from fractions import Fraction as F
from contribution_analysis.three_pool_t3_model_v1.fixtures import poly


def cases():
    out=[]
    bs=list(map(F,['2/3','4/5','9/10','7/10','5/6','19/20']))
    qs=list(map(F,['4/3','5/3','8/3','7/4','9/4','11/4']))
    for i,(b,q) in enumerate(zip(bs,qs)):
        entries=[('effective_information',[poly(1),poly(-b)],['3/4','1/4'],poly(1),1,1),
                 ('erroneous_information',[poly(1),poly(-b)],['3/4','1/4'],poly(-1),1,-1),
                 ('effective_restraint',[poly(1,2*q),poly(-1)],['1/2','1/2'],poly(0,q),1,1),
                 ('erroneous_block',[poly(1,2*q),poly(-1)],['1/2','1/2'],poly(-1),1,-1),
                 ('redundant_information',[poly(1),poly(1+b)],['1/2','1/2'],poly(1),0,0)]
        for cat,models,weights,ref,cs,ds in entries:
            out.append(dict(id=f'{cat}_{i}',category=cat,split='development' if i<3 else 'heldout_parameter',
                models=models,weights=weights,reference=ref,target=[0],anchor_weights=['1/2','1/2'],
                anchors=[['1','0'],['-1/4','0']],start=['0','0'],eta='1/10',expected_C_sign=cs,expected_D_sign=ds,
                parameter_index=i,b=str(b),q=str(q)))
    return out
