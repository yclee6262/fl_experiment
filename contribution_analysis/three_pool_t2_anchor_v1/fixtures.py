"""Rational fixtures frozen before running either development or held-out data."""
from fractions import Fraction as F


def cases():
    out=[]
    def add(category,index,l,Q,anchors,target,start=(0,0),expected_sign=0,expected_p=None):
        out.append(dict(id=category+'_'+str(index),category=category,split='development' if index<3 else 'heldout',
                        linear=[str(F(x)) for x in l],quadratic=[[str(F(x)) for x in row] for row in Q],
                        anchors=[[str(F(x)) for x in row] for row in anchors],start=[str(F(x)) for x in start],
                        target_owner=target,expected_C_sign=expected_sign,expected_p=expected_p,
                        offset='1000',eta='1/10',box=['-1','1'],mode='secant',direction_weights=['1/2','1/2']))
    grid=[('positive_helpful',[-1,-2,-3,F(-3,2),F(-5,2),F(-7,2)],0,1,'positive'),
          ('positive_harmful',[-14,-16,-18,-15,-17,-19],0,-1,'positive'),
          ('opposing_helpful',[F(-3,2),-2,F(-5,2),F(-7,4),F(-9,4),F(-11,4)],1,1,'negative'),
          ('opposing_harmful',[1,F(6,5),F(7,5),F(11,10),F(13,10),F(3,2)],1,-1,'negative')]
    for cat,ys,target,sign,p in grid:
        for i,y in enumerate(ys):
            qyy=(F(-2)-F(4,5)-F(3,5)*y-F(48,25))/F(9,25)
            add(cat,i,[1,y],[[3,0],[0,qyy]],[[1,0],[F(4,5),F(3,5)]],target,expected_sign=sign,expected_p=p)
    for i,c in enumerate([6,7,8,F(13,2),F(15,2),F(17,2)]):add('collinear_no_effect',i,[1,0],[[c,0],[0,0]],[[1,0],[F(-1,4),0]],1,expected_p='negative')
    for i,c in enumerate([1,2,3,F(3,2),F(5,2),F(7,2)]):add('rejected_equal',i,[0,0],[[c,0],[0,c]],[[1,0],[0,1]],0,expected_p='positive')
    for i,s in enumerate([F(96,100),F(97,100),F(98,100),F(965,1000),F(975,1000),F(985,1000)]):
        add('boundary',i,[-1,0],[[0,0],[0,0]],[[0,0],[s,1]],0,start=[s,0],expected_sign=1,expected_p='positive')
    for i,(l,e) in enumerate([(F(1),F(0)),(F(1),F(1,1000000)),(F(1),F(-1,1000000)),
                               (F(3,2),F(0)),(F(3,2),F(1,1000000)),(F(3,2),F(-1,1000000))]):
        add('cancellation',i,[l,0],[[F(-8,3)*l+e,0],[0,0]],[[1,0],[F(-1,4),0]],0,
            expected_sign=-1 if e<=0 else 0,expected_p='undefined' if e==0 else 'negative' if e>0 else 'positive')
    return out
