#include <STEPControl_Reader.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <TopoDS_Shape.hxx>
#include <TopExp_Explorer.hxx>
#include <TopAbs_ShapeEnum.hxx>
#include <GProp_GProps.hxx>
#include <BRepGProp.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <iostream>
#include <iomanip>

static int count(const TopoDS_Shape& s, TopAbs_ShapeEnum t) {
  int n=0; for (TopExp_Explorer ex(s,t); ex.More(); ex.Next()) ++n; return n;
}
int main(int argc,char**argv){
  if(argc!=2){std::cerr<<"usage: validator file.step\n";return 2;}
  STEPControl_Reader r;
  auto st=r.ReadFile(argv[1]);
  if(st!=IFSelect_RetDone){std::cerr<<"read_failed "<<int(st)<<"\n";return 3;}
  int roots=r.NbRootsForTransfer();
  int tr=r.TransferRoots();
  TopoDS_Shape s=r.OneShape();
  GProp_GProps v,a,l;
  BRepGProp::VolumeProperties(s,v);
  BRepGProp::SurfaceProperties(s,a);
  BRepGProp::LinearProperties(s,l);
  Bnd_Box b; BRepBndLib::Add(s,b,true);
  Standard_Real xmin,ymin,zmin,xmax,ymax,zmax; b.Get(xmin,ymin,zmin,xmax,ymax,zmax);
  std::cout<<std::setprecision(17);
  std::cout<<"roots="<<roots<<" transferred="<<tr<<"\n";
  std::cout<<"volume="<<v.Mass()<<" area="<<a.Mass()<<" length="<<l.Mass()<<"\n";
  std::cout<<"bbox="<<xmin<<","<<ymin<<","<<zmin<<","<<xmax<<","<<ymax<<","<<zmax<<"\n";
  std::cout<<"solids="<<count(s,TopAbs_SOLID)<<" shells="<<count(s,TopAbs_SHELL)
           <<" faces="<<count(s,TopAbs_FACE)<<" edges="<<count(s,TopAbs_EDGE)
           <<" vertices="<<count(s,TopAbs_VERTEX)<<"\n";
  std::cout<<"valid="<<(BRepCheck_Analyzer(s).IsValid()?1:0)<<"\n";
  return 0;
}
