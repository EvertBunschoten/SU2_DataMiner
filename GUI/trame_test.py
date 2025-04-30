from trame.app import get_server 
from trame.widgets import vuetify, paraview 
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, vtk as vtk_widgets
import vtk
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkColorTransferFunction,
)
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor, vtkScalarBarActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget, vtkScalarBarWidget
from Common.Properties import DefaultSettings_FGM
from Common.DataDrivenConfig import Config_FGM 

#from .pipeline import PipelineManager
from trame.assets.local import LocalFileManager

import numpy as np 
import time 

server = get_server(client_type='vue2')
renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
# offscreen rendering, no additional pop-up window
renderWindow.SetOffScreenRendering(1)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------
state, ctrl = server.state, server.controller

# State variables
state.config_filename = DefaultSettings_FGM.config_name + ".cfg"


# vtk named colors
colors = vtkNamedColors()

###################################
# ##### gradient background ##### #
###################################
# bottom: white
renderer.SetBackground(1., 1., 1.)
# top: light blue
renderer.SetBackground2(0.6, 0.8, 1.0)
# activate gradient background
renderer.GradientBackgroundOn()
###################################

flamelet_data_file = "/home/ecbunschoten/RACE/NonPremixed/flameletdata_pdiff/fluid_data_full.csv"
with open(flamelet_data_file,'r') as fid:
    variables = fid.readline().strip().split(',')

D = np.loadtxt(flamelet_data_file,delimiter=',',skiprows=1)
D_max, D_min = np.max(D,axis=0),np.min(D,axis=0)
nPoints = np.shape(D)[0]

mapper = vtk.vtkPolyDataMapper()
pointcloud_actor = vtk.vtkActor()

pointcloud = vtk.vtkPolyData()
CV_norm = (D[:, :3] - D_min[:3])/(D_max[:3] - D_min[:3])
vtkPoints = vtk.vtkPoints()
vtkCells = vtk.vtkCellArray()
vtkDepth = vtk.vtkDoubleArray()
vtkCells.InsertNextCell(1)
for iPoint in range(nPoints):
    pointId = vtkPoints.InsertNextPoint(CV_norm[iPoint, :])
    vtkCells.InsertNextCell(1)
    vtkCells.InsertCellPoint(pointId)
vtkCells.Modified()
vtkPoints.Modified()
vtkDepth.Modified()


pointcloud.SetPoints(vtkPoints)
pointcloud.SetVerts(vtkCells)
# Extract Array/Field information
datasetArrays = []

for ivar, var in enumerate(variables):
    name = var
    ArrayObject = vtk.vtkFloatArray()
    ArrayObject.SetName(name)
    # all components are scalars, no vectors for velocity
    ArrayObject.SetNumberOfComponents(1)
    # how many elements do we have?
    nElems = 1
    ArrayObject.SetNumberOfValues(nElems)
    ArrayObject.SetNumberOfTuples(nPoints)
    for i in range(nPoints):
        ArrayObject.SetValue(i, D[i, ivar])
    
    pointcloud.GetPointData().AddArray(ArrayObject)
    datasetArrays.append(
        {
            "text": name,
            "value": ivar,
            "range": [D_min[ivar],D_max[ivar]],
            "type": vtkDataObject.FIELD_ASSOCIATION_POINTS,
        }
    )

default_var = "Temperature"
default_array = datasetArrays[variables.index(default_var)]
pointcloud.GetPointData().SetScalars(pointcloud.GetPointData().GetArray(variables.index(default_var)))
pointcloud.GetPointData().SetActiveScalars(default_var)

default_min, default_max = default_array.get("range")
state.dataset_arrays = datasetArrays 

mapper.SetInputData(pointcloud)
mapper.SetColorMode(0)
mapper.SetScalarRange(default_min, default_max)
mapper.GetLookupTable().SetRange(default_min, default_max)
mapper.SetScalarVisibility(1)
mapper.SetScalarVisibility(True)


pointcloud_actor.SetMapper(mapper)
pointcloud_actor.SetObjectName("flamelet data point cloud")
pointcloud_actor.GetProperty().SetPointSize(4)
pointcloud_actor.GetProperty().SetRepresentationToPoints()

renderer.AddActor(pointcloud_actor)

renderWindow.AddRenderer(renderer)

DEFAULT_Z=0.5

from vtk import vtkPlaneSource
planeSource = vtk.vtkPlaneSource()
planeSource.SetCenter(0.5,0.5,0.5)
planeSource.SetNormal(0,0,1.0)
planeSource.Update()
plane = planeSource.GetOutput()
plane_mapper = vtk.vtkPolyDataMapper()
plane_mapper.SetInputData(plane)
plane_actor = vtk.vtkActor()
plane_actor.SetMapper(plane_mapper)

renderer.AddActor(plane_actor)
def MakeScalarBarActor(varname):
    # Create a scalar bar
    scalarbar = vtkScalarBarActor()
    scalarbar.SetObjectName("ScalarAxes")

    scalarbar.SetLookupTable(mapper.GetLookupTable())
    scalarbar.SetTitle(varname)
    scalarbar.SetVerticalTitleSeparation(10)
    scalarbar.UnconstrainedFontSizeOn()
    scalarbar.SetBarRatio(0.2)
    #scalar_bar.SetNumberOfLabels(5)
    scalarbar.SetMaximumWidthInPixels(100)
    scalarbar.SetMaximumHeightInPixels(600)
    return scalarbar

def MakeScalarBarWidget(scalarbar):
    # create the scalar_bar_widget
    scalarbarwidget = vtkScalarBarWidget()
    scalarbarwidget.SetInteractor(renderWindowInteractor)
    scalarbarwidget.SetScalarBarActor(scalarbar)
    scalarbarwidget.RepositionableOn()
    scalarbarwidget.On()
    return scalarbarwidget

scalar_bar = MakeScalarBarActor(default_var)
scalar_bar_widget =MakeScalarBarWidget(scalar_bar)

renderer.ResetCamera()


def VisualizeVariable(varname):
    data_array = datasetArrays[variables.index(varname)]
    pointcloud.GetPointData().SetActiveScalars(varname)

    data_min, data_max = data_array.get("range")
    state.dataset_arrays = datasetArrays 

    mapper.SetScalarRange(data_min, data_max)
    mapper.GetLookupTable().SetRange(data_min, data_max)

    scalar_bar.SetTitle(varname)
    ctrl.view_update()


@state.change("val_z_level")
def SetPlaneZ(val_z_level, **kwargs):
    planeSource.SetCenter(0.5,0.5,val_z_level)
    planeSource.Update()
    ctrl.view_update()

def reset_table_level():
    state.val_z_level=DEFAULT_Z

@state.change("idx_var_to_plot")
def update_mesh_color_by_name(idx_var_to_plot, **kwargs):
    print(idx_var_to_plot)
    VisualizeVariable(variables[idx_var_to_plot])
    ctrl.view_update()


def standard_buttons():
    vuetify.VCheckbox(
        v_model=("cube_axes_visibility", True),
        on_icon="mdi-cube-outline",
        off_icon="mdi-cube-off-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model="$vuetify.theme.dark",
        on_icon="mdi-lightbulb-off-outline",
        off_icon="mdi-lightbulb-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model=("viewMode", "local"),
        on_icon="mdi-lan-disconnect",
        off_icon="mdi-lan-connect",
        true_value="local",
        false_value="remote",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click="$refs.view.resetCamera()"):
        vuetify.VIcon("mdi-crop-free")

with SinglePageLayout(server) as layout:
    
    layout.title.set_text("Test GUI for SU2 DataMiner")
    #
    with layout.toolbar:
        # toolbar components
        vuetify.VSpacer()
        vuetify.VDivider(vertical=True, classes="mx-2")
        standard_buttons()

        vuetify.VSpacer()
        vuetify.VSlider(v_model=("val_z_level", DEFAULT_Z),
                        min=0,
                        max=1.0,
                        step=1e-3,
                        hide_details=True,
                        dense=True,
                        style="max-width: 300px")
        ######################################################
        # scalar selection field inside the top toolbar
        vuetify.VSelect(
            # Color By
            label="Color by",
            v_model=("idx_var_to_plot", 0),
            #items=("array_list", datasetArrays),
            items=("Object.values(dataset_arrays)",),
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1",
            #v_model=("field", "solid"),
            #items=("Object.values(fields)",),
            style="max-width: 300px;",
            #classes="mr-4",
        )
        ######################################################
        
    # content components
    with layout.content:
        with vuetify.VContainer(fluid=True,classes="pa-0 fill-height",):
            view = vtk_widgets.VtkRemoteView(renderWindow, interactive_ratio=1)
            # view = vtk.VtkLocalView(renderWindow)
            # view = vtk.VtkRemoteLocalView(
            #     renderWindow, namespace="view", mode="local", interactive_ratio=1
            # )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            ctrl.on_server_ready.add(view.update)
if __name__ == "__main__":
    server.start()
    