// Gazebo Harmonic native, synchronous simulation-only worker.
// The robot is never moved by setting its base pose/velocity during rollout.
// Commands drive the official left/right wheel joints through a bounded PI
// torque model. Calibration is necessary: this is not OpenCR firmware.
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/System.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/laserscan.pb.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using json=nlohmann::json;
using namespace gz::sim;
namespace c=gz::sim::components;
using Pair=std::array<double,2>;
using Pose=std::array<double,3>;
constexpr double pi=3.14159265358979323846;

class Drive final: public System, public ISystemPreUpdate, public ISystemPostUpdate {
 public:
  explicit Drive(const json &cfg): cfg(cfg) {
    const auto &m=cfg.at("motor");
    tau=m.at("tau_s"); accel=m.at("accel_rad_s2"); brake=m.at("brake_rad_s2");
    gain=m.at("gain"); dead=m.at("deadband_rad_s");
    kp=m.at("velocity_kp"); ki=m.at("velocity_ki"); torque=m.at("torque_limit_nm");
    start=cfg.at("start"); odom=start;
    dt=cfg.at("dt");
    delay=std::llround(m.at("delay_s").get<double>()/dt);
    queue.assign(delay+1,Pair{0,0});
    const size_t n=cfg.at("humans").get<size_t>();
    humanCmd.resize(n); humans.resize(n);
    for(size_t i=0;i<n;++i) humanCmd[i]={0,0};
  }
  void command(double v,double w,const std::vector<Pair>& hv,bool stop) {
    if (!std::isfinite(v)||!std::isfinite(w)||v<0||v>.100001||std::abs(w)>.500001||hv.size()!=humanCmd.size())
      throw std::runtime_error("invalid simulator command");
    for(auto a:hv) if(!std::isfinite(a[0])||!std::isfinite(a[1])||std::hypot(a[0],a[1])>2.01)
      throw std::runtime_error("invalid human command");
    requested={(v-.1435*w)/.033,(v+.1435*w)/.033};
    double scale=std::max({1.,std::abs(requested[0])/(.26/.033),std::abs(requested[1])/(.26/.033)});
    for(auto &x:requested)x/=scale;
    humanCmd=hv;
    if(stop) {
      requested={0,0};
      std::fill(queue.begin(),queue.end(),Pair{0,0});
    }
    contact=false; path=turn=0;
  }
  void PreUpdate(const UpdateInfo &info,EntityComponentManager &ecm) override {
    if(!ready) {
      robot=ecm.EntityByComponents(c::Model(),c::Name("turtlebot3_waffle_pi"));
      floor=ecm.EntityByComponents(c::Model(),c::Name("floor"));
      if(robot==kNullEntity||floor==kNullEntity)throw std::runtime_error("missing official robot/floor");
      Model model(robot);
      joints={model.JointByName(ecm,"wheel_left_joint"),model.JointByName(ecm,"wheel_right_joint")};
      for(auto j:joints) {
        if(j==kNullEntity)throw std::runtime_error("missing Waffle wheel joint");
        if(!ecm.Component<c::JointPosition>(j))ecm.CreateComponent(j,c::JointPosition());
        if(!ecm.Component<c::JointVelocity>(j))ecm.CreateComponent(j,c::JointVelocity());
      }
      for(size_t i=0;i<humans.size();++i) {
        humans[i]=ecm.EntityByComponents(c::Model(),c::Name("person_"+std::to_string(i)));
        if(humans[i]==kNullEntity)throw std::runtime_error("missing pedestrian model");
      }
      ecm.Each<c::Collision>([&](const Entity &e,const c::Collision*){
        if(descendant(e,robot,ecm)) robotCollisions.push_back(e);
        return true;
      });
      if(robotCollisions.empty())throw std::runtime_error("missing robot collision geometry");
      for(auto e:robotCollisions)
        if(!ecm.Component<c::ContactSensorData>(e))ecm.CreateComponent(e,c::ContactSensorData());
      ready=true;
    }
    if(info.paused)return;
    double step=std::chrono::duration<double>(info.dt).count();
    if(std::abs(step-dt)>1e-7)throw std::runtime_error("physics timestep mismatch");
    queue.push_back(requested);
    Pair target=queue.front();queue.pop_front();
    // Queue length equals delay+1: use current input for zero delay.
    if(delay==0)target=requested;
    else target=queue.front();
    for(size_t i=0;i<2;++i) {
      auto measured=ecm.Component<c::JointVelocity>(joints[i]);
      double actual=measured&&!measured->Data().empty()?measured->Data()[0]:0.;
      double desired=std::abs(target[i])<dead[i]?0.:target[i]*gain[i];
      bool slowing=reference[i]*desired<=0||std::abs(desired)<std::abs(reference[i]);
      double bound=(slowing?brake[i]:accel[i])*step;
      reference[i]+=std::clamp(-std::expm1(-step/tau[i])*(desired-reference[i]),-bound,bound);
      double error=reference[i]-actual;
      double candidate=integral[i]+ki[i]*error*step;
      double effort=kp[i]*error+candidate;
      if(std::abs(effort)<=torque[i]||effort*error<0)integral[i]=std::clamp(candidate,-torque[i],torque[i]);
      effort=std::clamp(kp[i]*error+integral[i],-torque[i],torque[i]);
      ecm.SetComponentData<c::JointForceCmd>(joints[i],{effort});
    }
    // Pedestrians are physically collidable simplified bodies; prescribed
    // planar crowd velocity is intentional, not robot actuation.
    for(size_t i=0;i<humans.size();++i) {
      auto p=worldPose(humans[i],ecm);
      double h=p.Rot().Yaw(),x=humanCmd[i][0],y=humanCmd[i][1];
      ecm.SetComponentData<c::LinearVelocityCmd>(humans[i],
          gz::math::Vector3d(std::cos(h)*x+std::sin(h)*y,-std::sin(h)*x+std::cos(h)*y,0));
      ecm.SetComponentData<c::AngularVelocityCmd>(humans[i],gz::math::Vector3d::Zero);
    }
  }
  void PostUpdate(const UpdateInfo &info,const EntityComponentManager &ecm) override {
    if(info.paused||!ready)return;
    simTime=std::chrono::duration<double>(info.simTime).count();
    auto p=worldPose(robot,ecm);
    Pose next={p.Pos().X(),p.Pos().Y(),p.Rot().Yaw()};
    if(haveTruth) {
      path+=std::hypot(next[0]-truth[0],next[1]-truth[1]);
      turn+=std::abs(std::remainder(next[2]-truth[2],2*pi));
    }
    truth=next;haveTruth=true;
    if(std::abs(p.Rot().Roll())>.25||std::abs(p.Rot().Pitch())>.25)contact=true;
    Pair enc;
    for(size_t i=0;i<2;++i) {
      auto q=ecm.Component<c::JointPosition>(joints[i]);
      auto v=ecm.Component<c::JointVelocity>(joints[i]);
      if(!q||q->Data().empty()||!v||v->Data().empty())return;
      wheels[i]=v->Data()[0];
      // 12-bit encoder position increments; nominal firmware geometry.
      enc[i]=std::round(q->Data()[0]*4096/(2*pi))*2*pi/4096;
    }
    if(haveEncoder) {
      double l=(enc[0]-previousEncoder[0])*.033,r=(enc[1]-previousEncoder[1])*.033;
      double distance=(l+r)/2,angle=(r-l)/.287;
      double scale=std::abs(angle)<1e-8?1.:std::sin(angle/2)/(angle/2);
      odom[0]+=distance*scale*std::cos(odom[2]+angle/2);
      odom[1]+=distance*scale*std::sin(odom[2]+angle/2);
      odom[2]=std::remainder(odom[2]+angle,2*pi);
    }
    previousEncoder=enc;haveEncoder=true;
    odomHistory.push_back({simTime,odom});
    while(odomHistory.size()>2&&odomHistory.front().first<simTime-2.)odomHistory.pop_front();
    humanPose.clear();
    for(auto h:humans) {
      auto q=worldPose(h,ecm);
      humanPose.push_back({q.Pos().X(),q.Pos().Y()});
    }
    for(auto e:robotCollisions) {
      auto data=ecm.Component<c::ContactSensorData>(e);
      if(!data)continue;
      for(const auto &hit:data->Data().contact()) {
        auto a=hit.collision1().id(),b=hit.collision2().id();
        if(descendant(a,robot,ecm)&&descendant(b,robot,ecm))continue;
        if(descendant(a,floor,ecm)||descendant(b,floor,ecm)) {
          floorContact=true;
          continue;
        }
        contact=true;
      }
    }
  }
  Pose poseAt(double t) const {
    if(odomHistory.empty()||t<odomHistory.front().first-dt||t>simTime+dt)
      throw std::runtime_error("scan outside odometry history");
    auto it=std::lower_bound(odomHistory.begin(),odomHistory.end(),t,
        [](const auto &p,double t){return p.first<t;});
    if(it==odomHistory.end())return odomHistory.back().second;
    if(it==odomHistory.begin())return it->second;
    auto a=std::prev(it);
    double k=(t-a->first)/(it->first-a->first);
    Pose p={a->second[0]+k*(it->second[0]-a->second[0]),a->second[1]+k*(it->second[1]-a->second[1]),
        a->second[2]+k*std::remainder(it->second[2]-a->second[2],2*pi)};
    return p;
  }
  static bool descendant(Entity e,Entity ancestor,const EntityComponentManager &ecm) {
    for(int k=0;k<32&&e!=kNullEntity;++k) {
      if(e==ancestor)return true;
      auto p=ecm.Component<c::ParentEntity>(e);
      if(!p)return false;
      e=p->Data();
    }
    return false;
  }
  json state()const {
    if(!haveEncoder||!haveTruth||humanPose.size()!=humans.size())throw std::runtime_error("missing physics feedback");
    return {{"time",simTime},{"truth",truth},{"odom",odom},{"wheels",wheels},
      {"twist",Pair{.033*(wheels[0]+wheels[1])/2,.033*(wheels[1]-wheels[0])/.287}},
      {"humans",humanPose},{"contact",contact},{"floor_contact_observed",floorContact},
      {"path",path},{"turn",turn}};
  }
  double simTime=0.,path=0.,turn=0.;
  bool contact=false,floorContact=false;
 private:
  json cfg;
  Entity robot=kNullEntity,floor=kNullEntity;
  std::array<Entity,2> joints{};
  std::vector<Entity> humans,robotCollisions;
  std::vector<Pair> humanCmd,humanPose;
  Pair tau,accel,brake,gain,dead,kp,ki,torque,requested{},reference{},integral{},wheels{},previousEncoder{};
  Pose truth{},odom{},start{};
  double dt;
  size_t delay;
  bool ready=false,haveTruth=false,haveEncoder=false;
  std::deque<Pair> queue;
  std::deque<std::pair<double,Pose>> odomHistory;
};

class Scans {
 public:
  void receive(const gz::msgs::LaserScan &scan) {
    std::lock_guard<std::mutex> lock(mu);
    double t=stamp(scan);
    if(frames.empty()||t>stamp(frames.back()))frames.push_back(scan);
    while(frames.size()>256)frames.pop_front();
    cv.notify_all();
  }
  static double stamp(const gz::msgs::LaserScan &s) {
    return s.header().stamp().sec()+s.header().stamp().nsec()*1e-9;
  }
  json collect(const Drive &drive,double period,double dt) {
    std::unique_lock<std::mutex> lock(mu);
    auto valid=[&]{return std::max(last,frames.empty()?-1.:stamp(frames.back()))>=drive.simTime-period-2*dt-1e-7;};
    if(!cv.wait_for(lock,std::chrono::seconds(10),valid))
      throw std::runtime_error("no timely Gazebo laser frame");
    json result=json::array();
    while(!frames.empty()&&stamp(frames.front())<=drive.simTime+dt) {
      auto s=std::move(frames.front());frames.pop_front();
      double t=stamp(s);
      if(t<=last)continue;
      last=t;
      json ranges=json::array();
      for(auto r:s.ranges())ranges.push_back(std::isfinite(r)?json(r):json(nullptr));
      result.push_back({{"stamp",t},{"pose",drive.poseAt(t)},{"ranges",ranges},
        {"min",s.range_min()},{"max",s.range_max()},{"angle_min",s.angle_min()},
        {"angle_max",s.angle_max()},{"angle_step",s.angle_step()}});
    }
    return result;
  }
 private:
  std::mutex mu;
  std::condition_variable cv;
  std::deque<gz::msgs::LaserScan> frames;
  double last=-1.;
};

void respond(const json& j){std::cout<<"WAFFLE_FRAME "<<j.dump()<<std::endl;}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: waffle_gazebo_worker world.sdf config.json");
    std::ifstream in(argv[2]);json cfg;in>>cfg;
    ServerConfig sc;sc.SetSdfFile(argv[1]);sc.SetHeadlessRendering(true);
    Scans scans;
    gz::transport::Node node;
    if(!node.Subscribe("/crowdnav_sim/scan",&Scans::receive,&scans))throw std::runtime_error("laser subscription failed");
    Server server(sc);
    auto drive=std::make_shared<Drive>(cfg);
    if(!server.AddSystem(drive).value_or(false))throw std::runtime_error("cannot add drive system");
    std::string line;
    while(std::getline(std::cin,line)) {
      auto command=json::parse(line);
      if(command.at("op")=="close")break;
      if(command.at("op")!="step")throw std::runtime_error("unknown operation");
      auto n=command.at("iterations").get<uint64_t>();
      if(n==0||n>20000)throw std::runtime_error("invalid iteration count");
      drive->command(command.at("v"),command.at("w"),command.at("humans"),command.at("stop"));
      double before=drive->simTime;
      if(!server.Run(true,n,false))throw std::runtime_error("Gazebo Run failed");
      double dt=cfg.at("dt");
      if(std::abs(drive->simTime-before-n*dt)>2*dt+1e-7)throw std::runtime_error("unexpected physics interval");
      auto state=drive->state();
      state["scans"]=scans.collect(*drive,1./cfg.at("laser_hz").get<double>(),dt);
      state["sequence"]=command.at("sequence");
      respond(state);
    }
    return 0;
  } catch(const std::exception &e) {
    respond({{"error",e.what()}});
    return 1;
  }
}
