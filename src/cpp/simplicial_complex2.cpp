#include "mathutils/mesh/simplicial_complex2.hpp"
#include "mathutils/mesh/mesh_common.hpp"

namespace mathutils {
namespace mesh {

/////////////////////////////////////////////
/////////////////////////////////////////////
// SimplicialTopology2 //////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

MeshSamples SimplicialTopology2::to_mesh_samples() const {
  MeshSamples ms;
  ms["V_cycle_E"] = V_cycle_E;
  ms["V_cycle_F"] = V_cycle_F;
  return ms;
}

std::map<std::string_view, SamplesIndex>
SimplicialTopology2::to_topo_samples() const {
  std::map<std::string_view, SamplesIndex> ms;

  ms["V_cycle_E"] = V_cycle_E;
  ms["V_cycle_F"] = V_cycle_F;
  return ms;
}

void SimplicialTopology2::from_mesh_samples(MeshSamples &ms) {

  // V_cycle_E_ = std::get<SamplesIndex>(ms.at("V_cycle_E"));
  // ms.erase("V_cycle_E");
  // V_cycle_F_ = std::get<SamplesIndex>(ms.at("V_cycle_F"));
  // ms.erase("V_cycle_E");
  pop_variant_to_mat_from_mesh_samples("V_cycle_E", V_cycle_E, ms);
  pop_variant_to_mat_from_mesh_samples("V_cycle_F", V_cycle_F, ms);
}

} // namespace mesh
} // namespace mathutils
