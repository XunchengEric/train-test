
N
spin_latticePlaceholder*
dtype0*$
shape:?????????


K
transpose/permConst*
dtype0*%
valueB"             
J
	transpose	Transposespin_latticetranspose/perm*
Tperm0*
T0
?
8MC/conv_1/kernel/kernel/Initializer/random_uniform/shapeConst*
dtype0**
_class 
loc:@MC/conv_1/kernel/kernel*%
valueB"             
?
6MC/conv_1/kernel/kernel/Initializer/random_uniform/minConst*
dtype0**
_class 
loc:@MC/conv_1/kernel/kernel*
valueB 2@D???Ե?
?
6MC/conv_1/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_1/kernel/kernel*
valueB 2@D???Ե?*
dtype0
?
@MC/conv_1/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_1/kernel/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@MC/conv_1/kernel/kernel*
dtype0*
seed2 *

seed 
?
6MC/conv_1/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_1/kernel/kernel/Initializer/random_uniform/max6MC/conv_1/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_1/kernel/kernel
?
6MC/conv_1/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_1/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_1/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_1/kernel/kernel
?
2MC/conv_1/kernel/kernel/Initializer/random_uniformAdd6MC/conv_1/kernel/kernel/Initializer/random_uniform/mul6MC/conv_1/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_1/kernel/kernel
?
MC/conv_1/kernel/kernel
VariableV2*
shared_name **
_class 
loc:@MC/conv_1/kernel/kernel*
dtype0*
	container *
shape: 
?
MC/conv_1/kernel/kernel/AssignAssignMC/conv_1/kernel/kernel2MC/conv_1/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@MC/conv_1/kernel/kernel
v
MC/conv_1/kernel/kernel/readIdentityMC/conv_1/kernel/kernel*
T0**
_class 
loc:@MC/conv_1/kernel/kernel
?
MC/conv_1/Conv2DConv2D	transposeMC/conv_1/kernel/kernel/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations

?
%MC/conv_1/bias/bias/Initializer/zerosConst*&
_class
loc:@MC/conv_1/bias/bias*
valueB 2        *
dtype0
?
MC/conv_1/bias/bias
VariableV2*
shape: *
shared_name *&
_class
loc:@MC/conv_1/bias/bias*
dtype0*
	container 
?
MC/conv_1/bias/bias/AssignAssignMC/conv_1/bias/bias%MC/conv_1/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*&
_class
loc:@MC/conv_1/bias/bias
j
MC/conv_1/bias/bias/readIdentityMC/conv_1/bias/bias*
T0*&
_class
loc:@MC/conv_1/bias/bias
K
MC/conv_1/addAddV2MC/conv_1/Conv2DMC/conv_1/bias/bias/read*
T0
I
MC/Reshape/shapeConst*!
valueB"????$       *
dtype0
M

MC/ReshapeReshapeMC/conv_1/addMC/Reshape/shape*
T0*
Tshape0
O
%MC/maxpool_1/MaxPool1d/ExpandDims/dimConst*
value	B :*
dtype0
w
!MC/maxpool_1/MaxPool1d/ExpandDims
ExpandDims
MC/Reshape%MC/maxpool_1/MaxPool1d/ExpandDims/dim*

Tdim0*
T0
?
MC/maxpool_1/MaxPool1dMaxPool!MC/maxpool_1/MaxPool1d/ExpandDims*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*
T0
a
MC/maxpool_1/MaxPool1d/SqueezeSqueezeMC/maxpool_1/MaxPool1d*
squeeze_dims
*
T0
?
:MC/deconv_1/kernel/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@MC/deconv_1/kernel/kernel*!
valueB"          *
dtype0
?
8MC/deconv_1/kernel/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@MC/deconv_1/kernel/kernel*
valueB 2      п*
dtype0
?
8MC/deconv_1/kernel/kernel/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@MC/deconv_1/kernel/kernel*
valueB 2      ??
?
BMC/deconv_1/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_1/kernel/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel*
dtype0*
seed2 *

seed 
?
8MC/deconv_1/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_1/kernel/kernel/Initializer/random_uniform/max8MC/deconv_1/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel
?
8MC/deconv_1/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_1/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_1/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel
?
4MC/deconv_1/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_1/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_1/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel
?
MC/deconv_1/kernel/kernel
VariableV2*,
_class"
 loc:@MC/deconv_1/kernel/kernel*
dtype0*
	container *
shape: *
shared_name 
?
 MC/deconv_1/kernel/kernel/AssignAssignMC/deconv_1/kernel/kernel4MC/deconv_1/kernel/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel*
validate_shape(
|
MC/deconv_1/kernel/kernel/readIdentityMC/deconv_1/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel
S
MC/deconv_1/ShapeShapeMC/maxpool_1/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_1/strided_slice/stackConst*
valueB: *
dtype0
O
!MC/deconv_1/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_1/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_1/strided_sliceStridedSliceMC/deconv_1/ShapeMC/deconv_1/strided_slice/stack!MC/deconv_1/strided_slice/stack_1!MC/deconv_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
U
MC/deconv_1/Shape_1ShapeMC/maxpool_1/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_1/strided_slice_1/stackConst*
dtype0*
valueB:
Q
#MC/deconv_1/strided_slice_1/stack_1Const*
valueB:*
dtype0
Q
#MC/deconv_1/strided_slice_1/stack_2Const*
valueB:*
dtype0
?
MC/deconv_1/strided_slice_1StridedSliceMC/deconv_1/Shape_1!MC/deconv_1/strided_slice_1/stack#MC/deconv_1/strided_slice_1/stack_1#MC/deconv_1/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
;
MC/deconv_1/sub/yConst*
value	B :*
dtype0
O
MC/deconv_1/subSubMC/deconv_1/strided_slice_1MC/deconv_1/sub/y*
T0
;
MC/deconv_1/mul/yConst*
value	B :*
dtype0
C
MC/deconv_1/mulMulMC/deconv_1/subMC/deconv_1/mul/y*
T0
;
MC/deconv_1/add/yConst*
value	B :*
dtype0
E
MC/deconv_1/addAddV2MC/deconv_1/mulMC/deconv_1/add/y*
T0
U
+MC/deconv_1/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_1/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_1/MaxPool1d/Squeeze+MC/deconv_1/conv1d_transpose/ExpandDims/dim*

Tdim0*
T0
W
-MC/deconv_1/conv1d_transpose/ExpandDims_1/dimConst*
dtype0*
value	B : 
?
)MC/deconv_1/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_1/kernel/kernel/read-MC/deconv_1/conv1d_transpose/ExpandDims_1/dim*

Tdim0*
T0
m
,MC/deconv_1/conv1d_transpose/concat/values_0PackMC/deconv_1/strided_slice*
T0*

axis *
N
Z
,MC/deconv_1/conv1d_transpose/concat/values_1Const*
valueB:*
dtype0
X
.MC/deconv_1/conv1d_transpose/concat/values_2/1Const*
value	B :*
dtype0
?
,MC/deconv_1/conv1d_transpose/concat/values_2PackMC/deconv_1/add.MC/deconv_1/conv1d_transpose/concat/values_2/1*
T0*

axis *
N
R
(MC/deconv_1/conv1d_transpose/concat/axisConst*
dtype0*
value	B : 
?
#MC/deconv_1/conv1d_transpose/concatConcatV2,MC/deconv_1/conv1d_transpose/concat/values_0,MC/deconv_1/conv1d_transpose/concat/values_1,MC/deconv_1/conv1d_transpose/concat/values_2(MC/deconv_1/conv1d_transpose/concat/axis*
T0*
N*

Tidx0
?
MC/deconv_1/conv1d_transposeConv2DBackpropInput#MC/deconv_1/conv1d_transpose/concat)MC/deconv_1/conv1d_transpose/ExpandDims_1'MC/deconv_1/conv1d_transpose/ExpandDims*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
m
$MC/deconv_1/conv1d_transpose/SqueezeSqueezeMC/deconv_1/conv1d_transpose*
squeeze_dims
*
T0
?
'MC/deconv_1/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_1/bias/bias*
valueB2        *
dtype0
?
MC/deconv_1/bias/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@MC/deconv_1/bias/bias
?
MC/deconv_1/bias/bias/AssignAssignMC/deconv_1/bias/bias'MC/deconv_1/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@MC/deconv_1/bias/bias
p
MC/deconv_1/bias/bias/readIdentityMC/deconv_1/bias/bias*
T0*(
_class
loc:@MC/deconv_1/bias/bias
e
MC/deconv_1/add_1AddV2$MC/deconv_1/conv1d_transpose/SqueezeMC/deconv_1/bias/bias/read*
T0
O
MC/Reshape_1/shapeConst*
dtype0*%
valueB"????         
U
MC/Reshape_1ReshapeMC/deconv_1/add_1MC/Reshape_1/shape*
T0*
Tshape0
N
MC/Tile/multiplesConst*%
valueB"            *
dtype0
K
MC/TileTileMC/Reshape_1MC/Tile/multiples*

Tmultiples0*
T0
K
MC/Slice/beginConst*%
valueB"              *
dtype0
J
MC/Slice/sizeConst*%
valueB"????
   
   ????*
dtype0
O
MC/SliceSliceMC/TileMC/Slice/beginMC/Slice/size*
T0*
Index0
?
8MC/conv_2/kernel/kernel/Initializer/random_uniform/shapeConst*
dtype0**
_class 
loc:@MC/conv_2/kernel/kernel*%
valueB"            
?
6MC/conv_2/kernel/kernel/Initializer/random_uniform/minConst**
_class 
loc:@MC/conv_2/kernel/kernel*
valueB 2????????*
dtype0
?
6MC/conv_2/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_2/kernel/kernel*
valueB 2????????*
dtype0
?
@MC/conv_2/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_2/kernel/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
6MC/conv_2/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_2/kernel/kernel/Initializer/random_uniform/max6MC/conv_2/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
6MC/conv_2/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_2/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_2/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
2MC/conv_2/kernel/kernel/Initializer/random_uniformAdd6MC/conv_2/kernel/kernel/Initializer/random_uniform/mul6MC/conv_2/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
MC/conv_2/kernel/kernel
VariableV2*
shape:*
shared_name **
_class 
loc:@MC/conv_2/kernel/kernel*
dtype0*
	container 
?
MC/conv_2/kernel/kernel/AssignAssignMC/conv_2/kernel/kernel2MC/conv_2/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
v
MC/conv_2/kernel/kernel/readIdentityMC/conv_2/kernel/kernel*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
MC/conv_2/Conv2DConv2DMC/SliceMC/conv_2/kernel/kernel/read*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
?
%MC/conv_2/bias/bias/Initializer/zerosConst*&
_class
loc:@MC/conv_2/bias/bias*
valueB2        *
dtype0
?
MC/conv_2/bias/bias
VariableV2*
shared_name *&
_class
loc:@MC/conv_2/bias/bias*
dtype0*
	container *
shape:
?
MC/conv_2/bias/bias/AssignAssignMC/conv_2/bias/bias%MC/conv_2/bias/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@MC/conv_2/bias/bias*
validate_shape(
j
MC/conv_2/bias/bias/readIdentityMC/conv_2/bias/bias*
T0*&
_class
loc:@MC/conv_2/bias/bias
K
MC/conv_2/addAddV2MC/conv_2/Conv2DMC/conv_2/bias/bias/read*
T0
K
MC/Reshape_2/shapeConst*!
valueB"????$      *
dtype0
Q
MC/Reshape_2ReshapeMC/conv_2/addMC/Reshape_2/shape*
T0*
Tshape0
O
%MC/maxpool_2/MaxPool1d/ExpandDims/dimConst*
dtype0*
value	B :
y
!MC/maxpool_2/MaxPool1d/ExpandDims
ExpandDimsMC/Reshape_2%MC/maxpool_2/MaxPool1d/ExpandDims/dim*

Tdim0*
T0
?
MC/maxpool_2/MaxPool1dMaxPool!MC/maxpool_2/MaxPool1d/ExpandDims*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*
T0
a
MC/maxpool_2/MaxPool1d/SqueezeSqueezeMC/maxpool_2/MaxPool1d*
T0*
squeeze_dims

?
:MC/deconv_2/kernel/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@MC/deconv_2/kernel/kernel*!
valueB"         *
dtype0
?
8MC/deconv_2/kernel/kernel/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@MC/deconv_2/kernel/kernel*
valueB 2?;f??ֿ
?
8MC/deconv_2/kernel/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@MC/deconv_2/kernel/kernel*
valueB 2?;f????*
dtype0
?
BMC/deconv_2/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_2/kernel/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
?
8MC/deconv_2/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_2/kernel/kernel/Initializer/random_uniform/max8MC/deconv_2/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
?
8MC/deconv_2/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_2/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_2/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
?
4MC/deconv_2/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_2/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_2/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
?
MC/deconv_2/kernel/kernel
VariableV2*
shape:*
shared_name *,
_class"
 loc:@MC/deconv_2/kernel/kernel*
dtype0*
	container 
?
 MC/deconv_2/kernel/kernel/AssignAssignMC/deconv_2/kernel/kernel4MC/deconv_2/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
|
MC/deconv_2/kernel/kernel/readIdentityMC/deconv_2/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
S
MC/deconv_2/ShapeShapeMC/maxpool_2/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_2/strided_slice/stackConst*
dtype0*
valueB: 
O
!MC/deconv_2/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_2/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_2/strided_sliceStridedSliceMC/deconv_2/ShapeMC/deconv_2/strided_slice/stack!MC/deconv_2/strided_slice/stack_1!MC/deconv_2/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
U
MC/deconv_2/Shape_1ShapeMC/maxpool_2/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_2/strided_slice_1/stackConst*
valueB:*
dtype0
Q
#MC/deconv_2/strided_slice_1/stack_1Const*
valueB:*
dtype0
Q
#MC/deconv_2/strided_slice_1/stack_2Const*
valueB:*
dtype0
?
MC/deconv_2/strided_slice_1StridedSliceMC/deconv_2/Shape_1!MC/deconv_2/strided_slice_1/stack#MC/deconv_2/strided_slice_1/stack_1#MC/deconv_2/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
;
MC/deconv_2/sub/yConst*
value	B :*
dtype0
O
MC/deconv_2/subSubMC/deconv_2/strided_slice_1MC/deconv_2/sub/y*
T0
;
MC/deconv_2/mul/yConst*
value	B :*
dtype0
C
MC/deconv_2/mulMulMC/deconv_2/subMC/deconv_2/mul/y*
T0
;
MC/deconv_2/add/yConst*
value	B :*
dtype0
E
MC/deconv_2/addAddV2MC/deconv_2/mulMC/deconv_2/add/y*
T0
U
+MC/deconv_2/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_2/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_2/MaxPool1d/Squeeze+MC/deconv_2/conv1d_transpose/ExpandDims/dim*

Tdim0*
T0
W
-MC/deconv_2/conv1d_transpose/ExpandDims_1/dimConst*
value	B : *
dtype0
?
)MC/deconv_2/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_2/kernel/kernel/read-MC/deconv_2/conv1d_transpose/ExpandDims_1/dim*

Tdim0*
T0
m
,MC/deconv_2/conv1d_transpose/concat/values_0PackMC/deconv_2/strided_slice*
T0*

axis *
N
Z
,MC/deconv_2/conv1d_transpose/concat/values_1Const*
valueB:*
dtype0
X
.MC/deconv_2/conv1d_transpose/concat/values_2/1Const*
value	B :*
dtype0
?
,MC/deconv_2/conv1d_transpose/concat/values_2PackMC/deconv_2/add.MC/deconv_2/conv1d_transpose/concat/values_2/1*
N*
T0*

axis 
R
(MC/deconv_2/conv1d_transpose/concat/axisConst*
value	B : *
dtype0
?
#MC/deconv_2/conv1d_transpose/concatConcatV2,MC/deconv_2/conv1d_transpose/concat/values_0,MC/deconv_2/conv1d_transpose/concat/values_1,MC/deconv_2/conv1d_transpose/concat/values_2(MC/deconv_2/conv1d_transpose/concat/axis*

Tidx0*
T0*
N
?
MC/deconv_2/conv1d_transposeConv2DBackpropInput#MC/deconv_2/conv1d_transpose/concat)MC/deconv_2/conv1d_transpose/ExpandDims_1'MC/deconv_2/conv1d_transpose/ExpandDims*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
m
$MC/deconv_2/conv1d_transpose/SqueezeSqueezeMC/deconv_2/conv1d_transpose*
T0*
squeeze_dims

?
'MC/deconv_2/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_2/bias/bias*
valueB2        *
dtype0
?
MC/deconv_2/bias/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@MC/deconv_2/bias/bias
?
MC/deconv_2/bias/bias/AssignAssignMC/deconv_2/bias/bias'MC/deconv_2/bias/bias/Initializer/zeros*
T0*(
_class
loc:@MC/deconv_2/bias/bias*
validate_shape(*
use_locking(
p
MC/deconv_2/bias/bias/readIdentityMC/deconv_2/bias/bias*
T0*(
_class
loc:@MC/deconv_2/bias/bias
e
MC/deconv_2/add_1AddV2$MC/deconv_2/conv1d_transpose/SqueezeMC/deconv_2/bias/bias/read*
T0
O
MC/Reshape_3/shapeConst*%
valueB"????         *
dtype0
U
MC/Reshape_3ReshapeMC/deconv_2/add_1MC/Reshape_3/shape*
T0*
Tshape0
P
MC/Tile_1/multiplesConst*
dtype0*%
valueB"            
O
	MC/Tile_1TileMC/Reshape_3MC/Tile_1/multiples*

Tmultiples0*
T0
M
MC/Slice_1/beginConst*%
valueB"              *
dtype0
L
MC/Slice_1/sizeConst*%
valueB"????      ????*
dtype0
W

MC/Slice_1Slice	MC/Tile_1MC/Slice_1/beginMC/Slice_1/size*
T0*
Index0
?
8MC/conv_3/kernel/kernel/Initializer/random_uniform/shapeConst**
_class 
loc:@MC/conv_3/kernel/kernel*%
valueB"            *
dtype0
?
6MC/conv_3/kernel/kernel/Initializer/random_uniform/minConst**
_class 
loc:@MC/conv_3/kernel/kernel*
valueB 23?E?y¿*
dtype0
?
6MC/conv_3/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_3/kernel/kernel*
valueB 23?E?y??*
dtype0
?
@MC/conv_3/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_3/kernel/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0**
_class 
loc:@MC/conv_3/kernel/kernel
?
6MC/conv_3/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_3/kernel/kernel/Initializer/random_uniform/max6MC/conv_3/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_3/kernel/kernel
?
6MC/conv_3/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_3/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_3/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_3/kernel/kernel
?
2MC/conv_3/kernel/kernel/Initializer/random_uniformAdd6MC/conv_3/kernel/kernel/Initializer/random_uniform/mul6MC/conv_3/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_3/kernel/kernel
?
MC/conv_3/kernel/kernel
VariableV2*
shared_name **
_class 
loc:@MC/conv_3/kernel/kernel*
dtype0*
	container *
shape:
?
MC/conv_3/kernel/kernel/AssignAssignMC/conv_3/kernel/kernel2MC/conv_3/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0**
_class 
loc:@MC/conv_3/kernel/kernel
v
MC/conv_3/kernel/kernel/readIdentityMC/conv_3/kernel/kernel*
T0**
_class 
loc:@MC/conv_3/kernel/kernel
?
MC/conv_3/Conv2DConv2D
MC/Slice_1MC/conv_3/kernel/kernel/read*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
%MC/conv_3/bias/bias/Initializer/zerosConst*
dtype0*&
_class
loc:@MC/conv_3/bias/bias*
valueB2        
?
MC/conv_3/bias/bias
VariableV2*
shape:*
shared_name *&
_class
loc:@MC/conv_3/bias/bias*
dtype0*
	container 
?
MC/conv_3/bias/bias/AssignAssignMC/conv_3/bias/bias%MC/conv_3/bias/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@MC/conv_3/bias/bias*
validate_shape(
j
MC/conv_3/bias/bias/readIdentityMC/conv_3/bias/bias*
T0*&
_class
loc:@MC/conv_3/bias/bias
K
MC/conv_3/addAddV2MC/conv_3/Conv2DMC/conv_3/bias/bias/read*
T0
K
MC/Reshape_4/shapeConst*!
valueB"????$      *
dtype0
Q
MC/Reshape_4ReshapeMC/conv_3/addMC/Reshape_4/shape*
T0*
Tshape0
O
%MC/maxpool_3/MaxPool1d/ExpandDims/dimConst*
value	B :*
dtype0
y
!MC/maxpool_3/MaxPool1d/ExpandDims
ExpandDimsMC/Reshape_4%MC/maxpool_3/MaxPool1d/ExpandDims/dim*

Tdim0*
T0
?
MC/maxpool_3/MaxPool1dMaxPool!MC/maxpool_3/MaxPool1d/ExpandDims*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*
T0
a
MC/maxpool_3/MaxPool1d/SqueezeSqueezeMC/maxpool_3/MaxPool1d*
squeeze_dims
*
T0
?
:MC/deconv_3/kernel/kernel/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@MC/deconv_3/kernel/kernel*!
valueB"         
?
8MC/deconv_3/kernel/kernel/Initializer/random_uniform/minConst*
dtype0*,
_class"
 loc:@MC/deconv_3/kernel/kernel*
valueB 2.!	??ӿ
?
8MC/deconv_3/kernel/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@MC/deconv_3/kernel/kernel*
valueB 2.!	????*
dtype0
?
BMC/deconv_3/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_3/kernel/kernel/Initializer/random_uniform/shape*

seed *
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel*
dtype0*
seed2 
?
8MC/deconv_3/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_3/kernel/kernel/Initializer/random_uniform/max8MC/deconv_3/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
?
8MC/deconv_3/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_3/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_3/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
?
4MC/deconv_3/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_3/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_3/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
?
MC/deconv_3/kernel/kernel
VariableV2*,
_class"
 loc:@MC/deconv_3/kernel/kernel*
dtype0*
	container *
shape:*
shared_name 
?
 MC/deconv_3/kernel/kernel/AssignAssignMC/deconv_3/kernel/kernel4MC/deconv_3/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
|
MC/deconv_3/kernel/kernel/readIdentityMC/deconv_3/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
S
MC/deconv_3/ShapeShapeMC/maxpool_3/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_3/strided_slice/stackConst*
valueB: *
dtype0
O
!MC/deconv_3/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_3/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_3/strided_sliceStridedSliceMC/deconv_3/ShapeMC/deconv_3/strided_slice/stack!MC/deconv_3/strided_slice/stack_1!MC/deconv_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
U
MC/deconv_3/Shape_1ShapeMC/maxpool_3/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_3/strided_slice_1/stackConst*
valueB:*
dtype0
Q
#MC/deconv_3/strided_slice_1/stack_1Const*
valueB:*
dtype0
Q
#MC/deconv_3/strided_slice_1/stack_2Const*
valueB:*
dtype0
?
MC/deconv_3/strided_slice_1StridedSliceMC/deconv_3/Shape_1!MC/deconv_3/strided_slice_1/stack#MC/deconv_3/strided_slice_1/stack_1#MC/deconv_3/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
;
MC/deconv_3/sub/yConst*
value	B :*
dtype0
O
MC/deconv_3/subSubMC/deconv_3/strided_slice_1MC/deconv_3/sub/y*
T0
;
MC/deconv_3/mul/yConst*
value	B :*
dtype0
C
MC/deconv_3/mulMulMC/deconv_3/subMC/deconv_3/mul/y*
T0
;
MC/deconv_3/add/yConst*
value	B :*
dtype0
E
MC/deconv_3/addAddV2MC/deconv_3/mulMC/deconv_3/add/y*
T0
U
+MC/deconv_3/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_3/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_3/MaxPool1d/Squeeze+MC/deconv_3/conv1d_transpose/ExpandDims/dim*

Tdim0*
T0
W
-MC/deconv_3/conv1d_transpose/ExpandDims_1/dimConst*
value	B : *
dtype0
?
)MC/deconv_3/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_3/kernel/kernel/read-MC/deconv_3/conv1d_transpose/ExpandDims_1/dim*

Tdim0*
T0
m
,MC/deconv_3/conv1d_transpose/concat/values_0PackMC/deconv_3/strided_slice*
T0*

axis *
N
Z
,MC/deconv_3/conv1d_transpose/concat/values_1Const*
valueB:*
dtype0
X
.MC/deconv_3/conv1d_transpose/concat/values_2/1Const*
value	B :*
dtype0
?
,MC/deconv_3/conv1d_transpose/concat/values_2PackMC/deconv_3/add.MC/deconv_3/conv1d_transpose/concat/values_2/1*
N*
T0*

axis 
R
(MC/deconv_3/conv1d_transpose/concat/axisConst*
value	B : *
dtype0
?
#MC/deconv_3/conv1d_transpose/concatConcatV2,MC/deconv_3/conv1d_transpose/concat/values_0,MC/deconv_3/conv1d_transpose/concat/values_1,MC/deconv_3/conv1d_transpose/concat/values_2(MC/deconv_3/conv1d_transpose/concat/axis*

Tidx0*
T0*
N
?
MC/deconv_3/conv1d_transposeConv2DBackpropInput#MC/deconv_3/conv1d_transpose/concat)MC/deconv_3/conv1d_transpose/ExpandDims_1'MC/deconv_3/conv1d_transpose/ExpandDims*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
m
$MC/deconv_3/conv1d_transpose/SqueezeSqueezeMC/deconv_3/conv1d_transpose*
squeeze_dims
*
T0
?
'MC/deconv_3/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_3/bias/bias*
valueB2        *
dtype0
?
MC/deconv_3/bias/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@MC/deconv_3/bias/bias
?
MC/deconv_3/bias/bias/AssignAssignMC/deconv_3/bias/bias'MC/deconv_3/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@MC/deconv_3/bias/bias
p
MC/deconv_3/bias/bias/readIdentityMC/deconv_3/bias/bias*
T0*(
_class
loc:@MC/deconv_3/bias/bias
e
MC/deconv_3/add_1AddV2$MC/deconv_3/conv1d_transpose/SqueezeMC/deconv_3/bias/bias/read*
T0
O
MC/Reshape_5/shapeConst*%
valueB"????         *
dtype0
U
MC/Reshape_5ReshapeMC/deconv_3/add_1MC/Reshape_5/shape*
T0*
Tshape0
P
MC/Tile_2/multiplesConst*%
valueB"            *
dtype0
O
	MC/Tile_2TileMC/Reshape_5MC/Tile_2/multiples*
T0*

Tmultiples0
M
MC/Slice_2/beginConst*%
valueB"              *
dtype0
L
MC/Slice_2/sizeConst*%
valueB"????      ????*
dtype0
W

MC/Slice_2Slice	MC/Tile_2MC/Slice_2/beginMC/Slice_2/size*
T0*
Index0
?
8MC/conv_4/kernel/kernel/Initializer/random_uniform/shapeConst**
_class 
loc:@MC/conv_4/kernel/kernel*%
valueB"            *
dtype0
?
6MC/conv_4/kernel/kernel/Initializer/random_uniform/minConst**
_class 
loc:@MC/conv_4/kernel/kernel*
valueB 23?E?y¿*
dtype0
?
6MC/conv_4/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_4/kernel/kernel*
valueB 23?E?y??*
dtype0
?
@MC/conv_4/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_4/kernel/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@MC/conv_4/kernel/kernel*
dtype0*
seed2 *

seed 
?
6MC/conv_4/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_4/kernel/kernel/Initializer/random_uniform/max6MC/conv_4/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_4/kernel/kernel
?
6MC/conv_4/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_4/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_4/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_4/kernel/kernel
?
2MC/conv_4/kernel/kernel/Initializer/random_uniformAdd6MC/conv_4/kernel/kernel/Initializer/random_uniform/mul6MC/conv_4/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_4/kernel/kernel
?
MC/conv_4/kernel/kernel
VariableV2**
_class 
loc:@MC/conv_4/kernel/kernel*
dtype0*
	container *
shape:*
shared_name 
?
MC/conv_4/kernel/kernel/AssignAssignMC/conv_4/kernel/kernel2MC/conv_4/kernel/kernel/Initializer/random_uniform*
T0**
_class 
loc:@MC/conv_4/kernel/kernel*
validate_shape(*
use_locking(
v
MC/conv_4/kernel/kernel/readIdentityMC/conv_4/kernel/kernel*
T0**
_class 
loc:@MC/conv_4/kernel/kernel
?
MC/conv_4/Conv2DConv2D
MC/Slice_2MC/conv_4/kernel/kernel/read*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
%MC/conv_4/bias/bias/Initializer/zerosConst*&
_class
loc:@MC/conv_4/bias/bias*
valueB2        *
dtype0
?
MC/conv_4/bias/bias
VariableV2*
shared_name *&
_class
loc:@MC/conv_4/bias/bias*
dtype0*
	container *
shape:
?
MC/conv_4/bias/bias/AssignAssignMC/conv_4/bias/bias%MC/conv_4/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*&
_class
loc:@MC/conv_4/bias/bias
j
MC/conv_4/bias/bias/readIdentityMC/conv_4/bias/bias*
T0*&
_class
loc:@MC/conv_4/bias/bias
K
MC/conv_4/addAddV2MC/conv_4/Conv2DMC/conv_4/bias/bias/read*
T0
K
MC/Reshape_6/shapeConst*!
valueB"????$      *
dtype0
Q
MC/Reshape_6ReshapeMC/conv_4/addMC/Reshape_6/shape*
T0*
Tshape0
O
%MC/maxpool_4/MaxPool1d/ExpandDims/dimConst*
value	B :*
dtype0
y
!MC/maxpool_4/MaxPool1d/ExpandDims
ExpandDimsMC/Reshape_6%MC/maxpool_4/MaxPool1d/ExpandDims/dim*
T0*

Tdim0
?
MC/maxpool_4/MaxPool1dMaxPool!MC/maxpool_4/MaxPool1d/ExpandDims*
ksize
*
paddingVALID*
T0*
strides
*
data_formatNHWC
a
MC/maxpool_4/MaxPool1d/SqueezeSqueezeMC/maxpool_4/MaxPool1d*
squeeze_dims
*
T0
?
:MC/deconv_4/kernel/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@MC/deconv_4/kernel/kernel*!
valueB"         *
dtype0
?
8MC/deconv_4/kernel/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@MC/deconv_4/kernel/kernel*
valueB 2.!	??ӿ*
dtype0
?
8MC/deconv_4/kernel/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@MC/deconv_4/kernel/kernel*
valueB 2.!	????*
dtype0
?
BMC/deconv_4/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_4/kernel/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
?
8MC/deconv_4/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_4/kernel/kernel/Initializer/random_uniform/max8MC/deconv_4/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
?
8MC/deconv_4/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_4/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_4/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
?
4MC/deconv_4/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_4/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_4/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
?
MC/deconv_4/kernel/kernel
VariableV2*,
_class"
 loc:@MC/deconv_4/kernel/kernel*
dtype0*
	container *
shape:*
shared_name 
?
 MC/deconv_4/kernel/kernel/AssignAssignMC/deconv_4/kernel/kernel4MC/deconv_4/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
|
MC/deconv_4/kernel/kernel/readIdentityMC/deconv_4/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel
S
MC/deconv_4/ShapeShapeMC/maxpool_4/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_4/strided_slice/stackConst*
valueB: *
dtype0
O
!MC/deconv_4/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_4/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_4/strided_sliceStridedSliceMC/deconv_4/ShapeMC/deconv_4/strided_slice/stack!MC/deconv_4/strided_slice/stack_1!MC/deconv_4/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask 
U
MC/deconv_4/Shape_1ShapeMC/maxpool_4/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_4/strided_slice_1/stackConst*
valueB:*
dtype0
Q
#MC/deconv_4/strided_slice_1/stack_1Const*
dtype0*
valueB:
Q
#MC/deconv_4/strided_slice_1/stack_2Const*
dtype0*
valueB:
?
MC/deconv_4/strided_slice_1StridedSliceMC/deconv_4/Shape_1!MC/deconv_4/strided_slice_1/stack#MC/deconv_4/strided_slice_1/stack_1#MC/deconv_4/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
;
MC/deconv_4/sub/yConst*
value	B :*
dtype0
O
MC/deconv_4/subSubMC/deconv_4/strided_slice_1MC/deconv_4/sub/y*
T0
;
MC/deconv_4/mul/yConst*
value	B :*
dtype0
C
MC/deconv_4/mulMulMC/deconv_4/subMC/deconv_4/mul/y*
T0
;
MC/deconv_4/add/yConst*
dtype0*
value	B :
E
MC/deconv_4/addAddV2MC/deconv_4/mulMC/deconv_4/add/y*
T0
U
+MC/deconv_4/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_4/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_4/MaxPool1d/Squeeze+MC/deconv_4/conv1d_transpose/ExpandDims/dim*

Tdim0*
T0
W
-MC/deconv_4/conv1d_transpose/ExpandDims_1/dimConst*
value	B : *
dtype0
?
)MC/deconv_4/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_4/kernel/kernel/read-MC/deconv_4/conv1d_transpose/ExpandDims_1/dim*

Tdim0*
T0
m
,MC/deconv_4/conv1d_transpose/concat/values_0PackMC/deconv_4/strided_slice*
T0*

axis *
N
Z
,MC/deconv_4/conv1d_transpose/concat/values_1Const*
valueB:*
dtype0
X
.MC/deconv_4/conv1d_transpose/concat/values_2/1Const*
value	B :*
dtype0
?
,MC/deconv_4/conv1d_transpose/concat/values_2PackMC/deconv_4/add.MC/deconv_4/conv1d_transpose/concat/values_2/1*
T0*

axis *
N
R
(MC/deconv_4/conv1d_transpose/concat/axisConst*
value	B : *
dtype0
?
#MC/deconv_4/conv1d_transpose/concatConcatV2,MC/deconv_4/conv1d_transpose/concat/values_0,MC/deconv_4/conv1d_transpose/concat/values_1,MC/deconv_4/conv1d_transpose/concat/values_2(MC/deconv_4/conv1d_transpose/concat/axis*
N*

Tidx0*
T0
?
MC/deconv_4/conv1d_transposeConv2DBackpropInput#MC/deconv_4/conv1d_transpose/concat)MC/deconv_4/conv1d_transpose/ExpandDims_1'MC/deconv_4/conv1d_transpose/ExpandDims*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
	dilations

m
$MC/deconv_4/conv1d_transpose/SqueezeSqueezeMC/deconv_4/conv1d_transpose*
T0*
squeeze_dims

?
'MC/deconv_4/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_4/bias/bias*
valueB2        *
dtype0
?
MC/deconv_4/bias/bias
VariableV2*(
_class
loc:@MC/deconv_4/bias/bias*
dtype0*
	container *
shape:*
shared_name 
?
MC/deconv_4/bias/bias/AssignAssignMC/deconv_4/bias/bias'MC/deconv_4/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@MC/deconv_4/bias/bias
p
MC/deconv_4/bias/bias/readIdentityMC/deconv_4/bias/bias*
T0*(
_class
loc:@MC/deconv_4/bias/bias
e
MC/deconv_4/add_1AddV2$MC/deconv_4/conv1d_transpose/SqueezeMC/deconv_4/bias/bias/read*
T0
O
MC/Reshape_7/shapeConst*%
valueB"????         *
dtype0
U
MC/Reshape_7ReshapeMC/deconv_4/add_1MC/Reshape_7/shape*
T0*
Tshape0
P
MC/Tile_3/multiplesConst*%
valueB"            *
dtype0
O
	MC/Tile_3TileMC/Reshape_7MC/Tile_3/multiples*

Tmultiples0*
T0
M
MC/Slice_3/beginConst*
dtype0*%
valueB"              
L
MC/Slice_3/sizeConst*
dtype0*%
valueB"????      ????
W

MC/Slice_3Slice	MC/Tile_3MC/Slice_3/beginMC/Slice_3/size*
T0*
Index0
?
8MC/conv_5/kernel/kernel/Initializer/random_uniform/shapeConst*
dtype0**
_class 
loc:@MC/conv_5/kernel/kernel*%
valueB"            
?
6MC/conv_5/kernel/kernel/Initializer/random_uniform/minConst**
_class 
loc:@MC/conv_5/kernel/kernel*
valueB 23?E?y¿*
dtype0
?
6MC/conv_5/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_5/kernel/kernel*
valueB 23?E?y??*
dtype0
?
@MC/conv_5/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_5/kernel/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@MC/conv_5/kernel/kernel*
dtype0*
seed2 *

seed 
?
6MC/conv_5/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_5/kernel/kernel/Initializer/random_uniform/max6MC/conv_5/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_5/kernel/kernel
?
6MC/conv_5/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_5/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_5/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_5/kernel/kernel
?
2MC/conv_5/kernel/kernel/Initializer/random_uniformAdd6MC/conv_5/kernel/kernel/Initializer/random_uniform/mul6MC/conv_5/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_5/kernel/kernel
?
MC/conv_5/kernel/kernel
VariableV2*
shared_name **
_class 
loc:@MC/conv_5/kernel/kernel*
dtype0*
	container *
shape:
?
MC/conv_5/kernel/kernel/AssignAssignMC/conv_5/kernel/kernel2MC/conv_5/kernel/kernel/Initializer/random_uniform*
T0**
_class 
loc:@MC/conv_5/kernel/kernel*
validate_shape(*
use_locking(
v
MC/conv_5/kernel/kernel/readIdentityMC/conv_5/kernel/kernel*
T0**
_class 
loc:@MC/conv_5/kernel/kernel
?
MC/conv_5/Conv2DConv2D
MC/Slice_3MC/conv_5/kernel/kernel/read*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0
?
%MC/conv_5/bias/bias/Initializer/zerosConst*&
_class
loc:@MC/conv_5/bias/bias*
valueB2        *
dtype0
?
MC/conv_5/bias/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *&
_class
loc:@MC/conv_5/bias/bias
?
MC/conv_5/bias/bias/AssignAssignMC/conv_5/bias/bias%MC/conv_5/bias/bias/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@MC/conv_5/bias/bias*
validate_shape(
j
MC/conv_5/bias/bias/readIdentityMC/conv_5/bias/bias*
T0*&
_class
loc:@MC/conv_5/bias/bias
K
MC/conv_5/addAddV2MC/conv_5/Conv2DMC/conv_5/bias/bias/read*
T0
K
MC/Reshape_8/shapeConst*!
valueB"????$      *
dtype0
Q
MC/Reshape_8ReshapeMC/conv_5/addMC/Reshape_8/shape*
T0*
Tshape0
O
%MC/maxpool_5/MaxPool1d/ExpandDims/dimConst*
value	B :*
dtype0
y
!MC/maxpool_5/MaxPool1d/ExpandDims
ExpandDimsMC/Reshape_8%MC/maxpool_5/MaxPool1d/ExpandDims/dim*
T0*

Tdim0
?
MC/maxpool_5/MaxPool1dMaxPool!MC/maxpool_5/MaxPool1d/ExpandDims*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
a
MC/maxpool_5/MaxPool1d/SqueezeSqueezeMC/maxpool_5/MaxPool1d*
squeeze_dims
*
T0
?
:MC/deconv_5/kernel/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@MC/deconv_5/kernel/kernel*!
valueB"         *
dtype0
?
8MC/deconv_5/kernel/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@MC/deconv_5/kernel/kernel*
valueB 2.!	??ӿ*
dtype0
?
8MC/deconv_5/kernel/kernel/Initializer/random_uniform/maxConst*
dtype0*,
_class"
 loc:@MC/deconv_5/kernel/kernel*
valueB 2.!	????
?
BMC/deconv_5/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_5/kernel/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel*
dtype0*
seed2 *

seed 
?
8MC/deconv_5/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_5/kernel/kernel/Initializer/random_uniform/max8MC/deconv_5/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel
?
8MC/deconv_5/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_5/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_5/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel
?
4MC/deconv_5/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_5/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_5/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel
?
MC/deconv_5/kernel/kernel
VariableV2*
shared_name *,
_class"
 loc:@MC/deconv_5/kernel/kernel*
dtype0*
	container *
shape:
?
 MC/deconv_5/kernel/kernel/AssignAssignMC/deconv_5/kernel/kernel4MC/deconv_5/kernel/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel*
validate_shape(
|
MC/deconv_5/kernel/kernel/readIdentityMC/deconv_5/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel
S
MC/deconv_5/ShapeShapeMC/maxpool_5/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_5/strided_slice/stackConst*
valueB: *
dtype0
O
!MC/deconv_5/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_5/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_5/strided_sliceStridedSliceMC/deconv_5/ShapeMC/deconv_5/strided_slice/stack!MC/deconv_5/strided_slice/stack_1!MC/deconv_5/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
U
MC/deconv_5/Shape_1ShapeMC/maxpool_5/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_5/strided_slice_1/stackConst*
dtype0*
valueB:
Q
#MC/deconv_5/strided_slice_1/stack_1Const*
valueB:*
dtype0
Q
#MC/deconv_5/strided_slice_1/stack_2Const*
valueB:*
dtype0
?
MC/deconv_5/strided_slice_1StridedSliceMC/deconv_5/Shape_1!MC/deconv_5/strided_slice_1/stack#MC/deconv_5/strided_slice_1/stack_1#MC/deconv_5/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
;
MC/deconv_5/sub/yConst*
value	B :*
dtype0
O
MC/deconv_5/subSubMC/deconv_5/strided_slice_1MC/deconv_5/sub/y*
T0
;
MC/deconv_5/mul/yConst*
dtype0*
value	B :
C
MC/deconv_5/mulMulMC/deconv_5/subMC/deconv_5/mul/y*
T0
;
MC/deconv_5/add/yConst*
dtype0*
value	B :
E
MC/deconv_5/addAddV2MC/deconv_5/mulMC/deconv_5/add/y*
T0
U
+MC/deconv_5/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_5/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_5/MaxPool1d/Squeeze+MC/deconv_5/conv1d_transpose/ExpandDims/dim*
T0*

Tdim0
W
-MC/deconv_5/conv1d_transpose/ExpandDims_1/dimConst*
value	B : *
dtype0
?
)MC/deconv_5/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_5/kernel/kernel/read-MC/deconv_5/conv1d_transpose/ExpandDims_1/dim*

Tdim0*
T0
m
,MC/deconv_5/conv1d_transpose/concat/values_0PackMC/deconv_5/strided_slice*
T0*

axis *
N
Z
,MC/deconv_5/conv1d_transpose/concat/values_1Const*
dtype0*
valueB:
X
.MC/deconv_5/conv1d_transpose/concat/values_2/1Const*
dtype0*
value	B :
?
,MC/deconv_5/conv1d_transpose/concat/values_2PackMC/deconv_5/add.MC/deconv_5/conv1d_transpose/concat/values_2/1*
T0*

axis *
N
R
(MC/deconv_5/conv1d_transpose/concat/axisConst*
value	B : *
dtype0
?
#MC/deconv_5/conv1d_transpose/concatConcatV2,MC/deconv_5/conv1d_transpose/concat/values_0,MC/deconv_5/conv1d_transpose/concat/values_1,MC/deconv_5/conv1d_transpose/concat/values_2(MC/deconv_5/conv1d_transpose/concat/axis*
T0*
N*

Tidx0
?
MC/deconv_5/conv1d_transposeConv2DBackpropInput#MC/deconv_5/conv1d_transpose/concat)MC/deconv_5/conv1d_transpose/ExpandDims_1'MC/deconv_5/conv1d_transpose/ExpandDims*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
m
$MC/deconv_5/conv1d_transpose/SqueezeSqueezeMC/deconv_5/conv1d_transpose*
squeeze_dims
*
T0
?
'MC/deconv_5/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_5/bias/bias*
valueB2        *
dtype0
?
MC/deconv_5/bias/bias
VariableV2*
shape:*
shared_name *(
_class
loc:@MC/deconv_5/bias/bias*
dtype0*
	container 
?
MC/deconv_5/bias/bias/AssignAssignMC/deconv_5/bias/bias'MC/deconv_5/bias/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*(
_class
loc:@MC/deconv_5/bias/bias
p
MC/deconv_5/bias/bias/readIdentityMC/deconv_5/bias/bias*
T0*(
_class
loc:@MC/deconv_5/bias/bias
e
MC/deconv_5/add_1AddV2$MC/deconv_5/conv1d_transpose/SqueezeMC/deconv_5/bias/bias/read*
T0
O
MC/Reshape_9/shapeConst*
dtype0*%
valueB"????         
U
MC/Reshape_9ReshapeMC/deconv_5/add_1MC/Reshape_9/shape*
T0*
Tshape0
P
MC/Tile_4/multiplesConst*%
valueB"            *
dtype0
O
	MC/Tile_4TileMC/Reshape_9MC/Tile_4/multiples*
T0*

Tmultiples0
M
MC/Slice_4/beginConst*%
valueB"              *
dtype0
L
MC/Slice_4/sizeConst*%
valueB"????      ????*
dtype0
W

MC/Slice_4Slice	MC/Tile_4MC/Slice_4/beginMC/Slice_4/size*
T0*
Index0
?
8MC/conv_6/kernel/kernel/Initializer/random_uniform/shapeConst**
_class 
loc:@MC/conv_6/kernel/kernel*%
valueB"            *
dtype0
?
6MC/conv_6/kernel/kernel/Initializer/random_uniform/minConst**
_class 
loc:@MC/conv_6/kernel/kernel*
valueB 23?E?y¿*
dtype0
?
6MC/conv_6/kernel/kernel/Initializer/random_uniform/maxConst**
_class 
loc:@MC/conv_6/kernel/kernel*
valueB 23?E?y??*
dtype0
?
@MC/conv_6/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform8MC/conv_6/kernel/kernel/Initializer/random_uniform/shape*
T0**
_class 
loc:@MC/conv_6/kernel/kernel*
dtype0*
seed2 *

seed 
?
6MC/conv_6/kernel/kernel/Initializer/random_uniform/subSub6MC/conv_6/kernel/kernel/Initializer/random_uniform/max6MC/conv_6/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_6/kernel/kernel
?
6MC/conv_6/kernel/kernel/Initializer/random_uniform/mulMul@MC/conv_6/kernel/kernel/Initializer/random_uniform/RandomUniform6MC/conv_6/kernel/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@MC/conv_6/kernel/kernel
?
2MC/conv_6/kernel/kernel/Initializer/random_uniformAdd6MC/conv_6/kernel/kernel/Initializer/random_uniform/mul6MC/conv_6/kernel/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@MC/conv_6/kernel/kernel
?
MC/conv_6/kernel/kernel
VariableV2*
dtype0*
	container *
shape:*
shared_name **
_class 
loc:@MC/conv_6/kernel/kernel
?
MC/conv_6/kernel/kernel/AssignAssignMC/conv_6/kernel/kernel2MC/conv_6/kernel/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@MC/conv_6/kernel/kernel*
validate_shape(
v
MC/conv_6/kernel/kernel/readIdentityMC/conv_6/kernel/kernel*
T0**
_class 
loc:@MC/conv_6/kernel/kernel
?
MC/conv_6/Conv2DConv2D
MC/Slice_4MC/conv_6/kernel/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
?
%MC/conv_6/bias/bias/Initializer/zerosConst*&
_class
loc:@MC/conv_6/bias/bias*
valueB2        *
dtype0
?
MC/conv_6/bias/bias
VariableV2*
shared_name *&
_class
loc:@MC/conv_6/bias/bias*
dtype0*
	container *
shape:
?
MC/conv_6/bias/bias/AssignAssignMC/conv_6/bias/bias%MC/conv_6/bias/bias/Initializer/zeros*
T0*&
_class
loc:@MC/conv_6/bias/bias*
validate_shape(*
use_locking(
j
MC/conv_6/bias/bias/readIdentityMC/conv_6/bias/bias*
T0*&
_class
loc:@MC/conv_6/bias/bias
K
MC/conv_6/addAddV2MC/conv_6/Conv2DMC/conv_6/bias/bias/read*
T0
L
MC/Reshape_10/shapeConst*!
valueB"????$      *
dtype0
S
MC/Reshape_10ReshapeMC/conv_6/addMC/Reshape_10/shape*
T0*
Tshape0
O
%MC/maxpool_6/MaxPool1d/ExpandDims/dimConst*
value	B :*
dtype0
z
!MC/maxpool_6/MaxPool1d/ExpandDims
ExpandDimsMC/Reshape_10%MC/maxpool_6/MaxPool1d/ExpandDims/dim*

Tdim0*
T0
?
MC/maxpool_6/MaxPool1dMaxPool!MC/maxpool_6/MaxPool1d/ExpandDims*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*
T0
a
MC/maxpool_6/MaxPool1d/SqueezeSqueezeMC/maxpool_6/MaxPool1d*
squeeze_dims
*
T0
?
:MC/deconv_6/kernel/kernel/Initializer/random_uniform/shapeConst*
dtype0*,
_class"
 loc:@MC/deconv_6/kernel/kernel*!
valueB"         
?
8MC/deconv_6/kernel/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@MC/deconv_6/kernel/kernel*
valueB 2v?u??ڿ*
dtype0
?
8MC/deconv_6/kernel/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@MC/deconv_6/kernel/kernel*
valueB 2v?u????*
dtype0
?
BMC/deconv_6/kernel/kernel/Initializer/random_uniform/RandomUniformRandomUniform:MC/deconv_6/kernel/kernel/Initializer/random_uniform/shape*

seed *
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel*
dtype0*
seed2 
?
8MC/deconv_6/kernel/kernel/Initializer/random_uniform/subSub8MC/deconv_6/kernel/kernel/Initializer/random_uniform/max8MC/deconv_6/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
?
8MC/deconv_6/kernel/kernel/Initializer/random_uniform/mulMulBMC/deconv_6/kernel/kernel/Initializer/random_uniform/RandomUniform8MC/deconv_6/kernel/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
?
4MC/deconv_6/kernel/kernel/Initializer/random_uniformAdd8MC/deconv_6/kernel/kernel/Initializer/random_uniform/mul8MC/deconv_6/kernel/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
?
MC/deconv_6/kernel/kernel
VariableV2*
shared_name *,
_class"
 loc:@MC/deconv_6/kernel/kernel*
dtype0*
	container *
shape:
?
 MC/deconv_6/kernel/kernel/AssignAssignMC/deconv_6/kernel/kernel4MC/deconv_6/kernel/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
|
MC/deconv_6/kernel/kernel/readIdentityMC/deconv_6/kernel/kernel*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
S
MC/deconv_6/ShapeShapeMC/maxpool_6/MaxPool1d/Squeeze*
T0*
out_type0
M
MC/deconv_6/strided_slice/stackConst*
valueB: *
dtype0
O
!MC/deconv_6/strided_slice/stack_1Const*
valueB:*
dtype0
O
!MC/deconv_6/strided_slice/stack_2Const*
valueB:*
dtype0
?
MC/deconv_6/strided_sliceStridedSliceMC/deconv_6/ShapeMC/deconv_6/strided_slice/stack!MC/deconv_6/strided_slice/stack_1!MC/deconv_6/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
U
MC/deconv_6/Shape_1ShapeMC/maxpool_6/MaxPool1d/Squeeze*
T0*
out_type0
O
!MC/deconv_6/strided_slice_1/stackConst*
valueB:*
dtype0
Q
#MC/deconv_6/strided_slice_1/stack_1Const*
dtype0*
valueB:
Q
#MC/deconv_6/strided_slice_1/stack_2Const*
valueB:*
dtype0
?
MC/deconv_6/strided_slice_1StridedSliceMC/deconv_6/Shape_1!MC/deconv_6/strided_slice_1/stack#MC/deconv_6/strided_slice_1/stack_1#MC/deconv_6/strided_slice_1/stack_2*
end_mask *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
;
MC/deconv_6/sub/yConst*
dtype0*
value	B :
O
MC/deconv_6/subSubMC/deconv_6/strided_slice_1MC/deconv_6/sub/y*
T0
;
MC/deconv_6/mul/yConst*
value	B :*
dtype0
C
MC/deconv_6/mulMulMC/deconv_6/subMC/deconv_6/mul/y*
T0
;
MC/deconv_6/add/yConst*
dtype0*
value	B :
E
MC/deconv_6/addAddV2MC/deconv_6/mulMC/deconv_6/add/y*
T0
U
+MC/deconv_6/conv1d_transpose/ExpandDims/dimConst*
value	B :*
dtype0
?
'MC/deconv_6/conv1d_transpose/ExpandDims
ExpandDimsMC/maxpool_6/MaxPool1d/Squeeze+MC/deconv_6/conv1d_transpose/ExpandDims/dim*

Tdim0*
T0
W
-MC/deconv_6/conv1d_transpose/ExpandDims_1/dimConst*
dtype0*
value	B : 
?
)MC/deconv_6/conv1d_transpose/ExpandDims_1
ExpandDimsMC/deconv_6/kernel/kernel/read-MC/deconv_6/conv1d_transpose/ExpandDims_1/dim*
T0*

Tdim0
m
,MC/deconv_6/conv1d_transpose/concat/values_0PackMC/deconv_6/strided_slice*
T0*

axis *
N
Z
,MC/deconv_6/conv1d_transpose/concat/values_1Const*
valueB:*
dtype0
X
.MC/deconv_6/conv1d_transpose/concat/values_2/1Const*
value	B :*
dtype0
?
,MC/deconv_6/conv1d_transpose/concat/values_2PackMC/deconv_6/add.MC/deconv_6/conv1d_transpose/concat/values_2/1*
T0*

axis *
N
R
(MC/deconv_6/conv1d_transpose/concat/axisConst*
value	B : *
dtype0
?
#MC/deconv_6/conv1d_transpose/concatConcatV2,MC/deconv_6/conv1d_transpose/concat/values_0,MC/deconv_6/conv1d_transpose/concat/values_1,MC/deconv_6/conv1d_transpose/concat/values_2(MC/deconv_6/conv1d_transpose/concat/axis*
T0*
N*

Tidx0
?
MC/deconv_6/conv1d_transposeConv2DBackpropInput#MC/deconv_6/conv1d_transpose/concat)MC/deconv_6/conv1d_transpose/ExpandDims_1'MC/deconv_6/conv1d_transpose/ExpandDims*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
m
$MC/deconv_6/conv1d_transpose/SqueezeSqueezeMC/deconv_6/conv1d_transpose*
squeeze_dims
*
T0
?
'MC/deconv_6/bias/bias/Initializer/zerosConst*(
_class
loc:@MC/deconv_6/bias/bias*
valueB2        *
dtype0
?
MC/deconv_6/bias/bias
VariableV2*
dtype0*
	container *
shape:*
shared_name *(
_class
loc:@MC/deconv_6/bias/bias
?
MC/deconv_6/bias/bias/AssignAssignMC/deconv_6/bias/bias'MC/deconv_6/bias/bias/Initializer/zeros*
T0*(
_class
loc:@MC/deconv_6/bias/bias*
validate_shape(*
use_locking(
p
MC/deconv_6/bias/bias/readIdentityMC/deconv_6/bias/bias*
T0*(
_class
loc:@MC/deconv_6/bias/bias
e
MC/deconv_6/add_1AddV2$MC/deconv_6/conv1d_transpose/SqueezeMC/deconv_6/bias/bias/read*
T0
H
MC/Reshape_11/shapeConst*
valueB"????$   *
dtype0
W
MC/Reshape_11ReshapeMC/deconv_6/add_1MC/Reshape_11/shape*
T0*
Tshape0
C
MC/Prod/reduction_indicesConst*
value	B :*
dtype0
_
MC/ProdProdMC/Reshape_11MC/Prod/reduction_indices*

Tidx0*
	keep_dims( *
T0
;

MC/SqueezeSqueezeMC/Prod*
T0*
squeeze_dims
 
'
logitsIdentity
MC/Squeeze*
T0
?
initNoOp^MC/conv_1/bias/bias/Assign^MC/conv_1/kernel/kernel/Assign^MC/conv_2/bias/bias/Assign^MC/conv_2/kernel/kernel/Assign^MC/conv_3/bias/bias/Assign^MC/conv_3/kernel/kernel/Assign^MC/conv_4/bias/bias/Assign^MC/conv_4/kernel/kernel/Assign^MC/conv_5/bias/bias/Assign^MC/conv_5/kernel/kernel/Assign^MC/conv_6/bias/bias/Assign^MC/conv_6/kernel/kernel/Assign^MC/deconv_1/bias/bias/Assign!^MC/deconv_1/kernel/kernel/Assign^MC/deconv_2/bias/bias/Assign!^MC/deconv_2/kernel/kernel/Assign^MC/deconv_3/bias/bias/Assign!^MC/deconv_3/kernel/kernel/Assign^MC/deconv_4/bias/bias/Assign!^MC/deconv_4/kernel/kernel/Assign^MC/deconv_5/bias/bias/Assign!^MC/deconv_5/kernel/kernel/Assign^MC/deconv_6/bias/bias/Assign!^MC/deconv_6/kernel/kernel/Assign
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
?
save/SaveV2/tensor_namesConst*?
value?B?BMC/conv_1/bias/biasBMC/conv_1/kernel/kernelBMC/conv_2/bias/biasBMC/conv_2/kernel/kernelBMC/conv_3/bias/biasBMC/conv_3/kernel/kernelBMC/conv_4/bias/biasBMC/conv_4/kernel/kernelBMC/conv_5/bias/biasBMC/conv_5/kernel/kernelBMC/conv_6/bias/biasBMC/conv_6/kernel/kernelBMC/deconv_1/bias/biasBMC/deconv_1/kernel/kernelBMC/deconv_2/bias/biasBMC/deconv_2/kernel/kernelBMC/deconv_3/bias/biasBMC/deconv_3/kernel/kernelBMC/deconv_4/bias/biasBMC/deconv_4/kernel/kernelBMC/deconv_5/bias/biasBMC/deconv_5/kernel/kernelBMC/deconv_6/bias/biasBMC/deconv_6/kernel/kernel*
dtype0
w
save/SaveV2/shape_and_slicesConst*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesMC/conv_1/bias/biasMC/conv_1/kernel/kernelMC/conv_2/bias/biasMC/conv_2/kernel/kernelMC/conv_3/bias/biasMC/conv_3/kernel/kernelMC/conv_4/bias/biasMC/conv_4/kernel/kernelMC/conv_5/bias/biasMC/conv_5/kernel/kernelMC/conv_6/bias/biasMC/conv_6/kernel/kernelMC/deconv_1/bias/biasMC/deconv_1/kernel/kernelMC/deconv_2/bias/biasMC/deconv_2/kernel/kernelMC/deconv_3/bias/biasMC/deconv_3/kernel/kernelMC/deconv_4/bias/biasMC/deconv_4/kernel/kernelMC/deconv_5/bias/biasMC/deconv_5/kernel/kernelMC/deconv_6/bias/biasMC/deconv_6/kernel/kernel*&
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?BMC/conv_1/bias/biasBMC/conv_1/kernel/kernelBMC/conv_2/bias/biasBMC/conv_2/kernel/kernelBMC/conv_3/bias/biasBMC/conv_3/kernel/kernelBMC/conv_4/bias/biasBMC/conv_4/kernel/kernelBMC/conv_5/bias/biasBMC/conv_5/kernel/kernelBMC/conv_6/bias/biasBMC/conv_6/kernel/kernelBMC/deconv_1/bias/biasBMC/deconv_1/kernel/kernelBMC/deconv_2/bias/biasBMC/deconv_2/kernel/kernelBMC/deconv_3/bias/biasBMC/deconv_3/kernel/kernelBMC/deconv_4/bias/biasBMC/deconv_4/kernel/kernelBMC/deconv_5/bias/biasBMC/deconv_5/kernel/kernelBMC/deconv_6/bias/biasBMC/deconv_6/kernel/kernel*
dtype0
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*&
dtypes
2
?
save/AssignAssignMC/conv_1/bias/biassave/RestoreV2*
use_locking(*
T0*&
_class
loc:@MC/conv_1/bias/bias*
validate_shape(
?
save/Assign_1AssignMC/conv_1/kernel/kernelsave/RestoreV2:1*
use_locking(*
T0**
_class 
loc:@MC/conv_1/kernel/kernel*
validate_shape(
?
save/Assign_2AssignMC/conv_2/bias/biassave/RestoreV2:2*
use_locking(*
T0*&
_class
loc:@MC/conv_2/bias/bias*
validate_shape(
?
save/Assign_3AssignMC/conv_2/kernel/kernelsave/RestoreV2:3*
validate_shape(*
use_locking(*
T0**
_class 
loc:@MC/conv_2/kernel/kernel
?
save/Assign_4AssignMC/conv_3/bias/biassave/RestoreV2:4*
use_locking(*
T0*&
_class
loc:@MC/conv_3/bias/bias*
validate_shape(
?
save/Assign_5AssignMC/conv_3/kernel/kernelsave/RestoreV2:5*
use_locking(*
T0**
_class 
loc:@MC/conv_3/kernel/kernel*
validate_shape(
?
save/Assign_6AssignMC/conv_4/bias/biassave/RestoreV2:6*
validate_shape(*
use_locking(*
T0*&
_class
loc:@MC/conv_4/bias/bias
?
save/Assign_7AssignMC/conv_4/kernel/kernelsave/RestoreV2:7*
T0**
_class 
loc:@MC/conv_4/kernel/kernel*
validate_shape(*
use_locking(
?
save/Assign_8AssignMC/conv_5/bias/biassave/RestoreV2:8*
T0*&
_class
loc:@MC/conv_5/bias/bias*
validate_shape(*
use_locking(
?
save/Assign_9AssignMC/conv_5/kernel/kernelsave/RestoreV2:9*
use_locking(*
T0**
_class 
loc:@MC/conv_5/kernel/kernel*
validate_shape(
?
save/Assign_10AssignMC/conv_6/bias/biassave/RestoreV2:10*
T0*&
_class
loc:@MC/conv_6/bias/bias*
validate_shape(*
use_locking(
?
save/Assign_11AssignMC/conv_6/kernel/kernelsave/RestoreV2:11*
T0**
_class 
loc:@MC/conv_6/kernel/kernel*
validate_shape(*
use_locking(
?
save/Assign_12AssignMC/deconv_1/bias/biassave/RestoreV2:12*
T0*(
_class
loc:@MC/deconv_1/bias/bias*
validate_shape(*
use_locking(
?
save/Assign_13AssignMC/deconv_1/kernel/kernelsave/RestoreV2:13*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_1/kernel/kernel
?
save/Assign_14AssignMC/deconv_2/bias/biassave/RestoreV2:14*
use_locking(*
T0*(
_class
loc:@MC/deconv_2/bias/bias*
validate_shape(
?
save/Assign_15AssignMC/deconv_2/kernel/kernelsave/RestoreV2:15*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_2/kernel/kernel
?
save/Assign_16AssignMC/deconv_3/bias/biassave/RestoreV2:16*
use_locking(*
T0*(
_class
loc:@MC/deconv_3/bias/bias*
validate_shape(
?
save/Assign_17AssignMC/deconv_3/kernel/kernelsave/RestoreV2:17*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_3/kernel/kernel
?
save/Assign_18AssignMC/deconv_4/bias/biassave/RestoreV2:18*
use_locking(*
T0*(
_class
loc:@MC/deconv_4/bias/bias*
validate_shape(
?
save/Assign_19AssignMC/deconv_4/kernel/kernelsave/RestoreV2:19*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_4/kernel/kernel*
validate_shape(
?
save/Assign_20AssignMC/deconv_5/bias/biassave/RestoreV2:20*
T0*(
_class
loc:@MC/deconv_5/bias/bias*
validate_shape(*
use_locking(
?
save/Assign_21AssignMC/deconv_5/kernel/kernelsave/RestoreV2:21*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_5/kernel/kernel*
validate_shape(
?
save/Assign_22AssignMC/deconv_6/bias/biassave/RestoreV2:22*
T0*(
_class
loc:@MC/deconv_6/bias/bias*
validate_shape(*
use_locking(
?
save/Assign_23AssignMC/deconv_6/kernel/kernelsave/RestoreV2:23*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@MC/deconv_6/kernel/kernel
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
9
gradients/ShapeShapelogits*
T0*
out_type0
D
gradients/grad_ys_0Const*
valueB 2      ??*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
J
gradients/MC/Squeeze_grad/ShapeShapeMC/Prod*
T0*
out_type0
t
!gradients/MC/Squeeze_grad/ReshapeReshapegradients/Fillgradients/MC/Squeeze_grad/Shape*
T0*
Tshape0
M
gradients/MC/Prod_grad/ShapeShapeMC/Reshape_11*
T0*
out_type0
[
$gradients/MC/Prod_grad/Reshape/shapeConst*
valueB:
?????????*
dtype0
?
gradients/MC/Prod_grad/ReshapeReshapeMC/Prod/reduction_indices$gradients/MC/Prod_grad/Reshape/shape*
T0*
Tshape0
E
gradients/MC/Prod_grad/SizeConst*
value	B :*
dtype0
d
gradients/MC/Prod_grad/addAddV2MC/Prod/reduction_indicesgradients/MC/Prod_grad/Size*
T0
h
gradients/MC/Prod_grad/modFloorModgradients/MC/Prod_grad/addgradients/MC/Prod_grad/Size*
T0
G
gradients/MC/Prod_grad/Shape_1Const*
dtype0*
valueB 
L
"gradients/MC/Prod_grad/range/startConst*
value	B : *
dtype0
L
"gradients/MC/Prod_grad/range/deltaConst*
value	B :*
dtype0
?
gradients/MC/Prod_grad/rangeRange"gradients/MC/Prod_grad/range/startgradients/MC/Prod_grad/Size"gradients/MC/Prod_grad/range/delta*

Tidx0
K
!gradients/MC/Prod_grad/Fill/valueConst*
value	B :*
dtype0
?
gradients/MC/Prod_grad/FillFillgradients/MC/Prod_grad/Shape_1!gradients/MC/Prod_grad/Fill/value*
T0*

index_type0
?
$gradients/MC/Prod_grad/DynamicStitchDynamicStitchgradients/MC/Prod_grad/rangegradients/MC/Prod_grad/modgradients/MC/Prod_grad/Shapegradients/MC/Prod_grad/Fill*
T0*
N
J
 gradients/MC/Prod_grad/Maximum/yConst*
dtype0*
value	B :
z
gradients/MC/Prod_grad/MaximumMaximum$gradients/MC/Prod_grad/DynamicStitch gradients/MC/Prod_grad/Maximum/y*
T0
r
gradients/MC/Prod_grad/floordivFloorDivgradients/MC/Prod_grad/Shapegradients/MC/Prod_grad/Maximum*
T0
?
 gradients/MC/Prod_grad/Reshape_1Reshape!gradients/MC/Squeeze_grad/Reshape$gradients/MC/Prod_grad/DynamicStitch*
T0*
Tshape0
?
gradients/MC/Prod_grad/TileTile gradients/MC/Prod_grad/Reshape_1gradients/MC/Prod_grad/floordiv*
T0*

Tmultiples0
T
gradients/MC/Prod_grad/RankConst"/device:CPU:0*
value	B :*
dtype0
z
gradients/MC/Prod_grad/add_1AddV2gradients/MC/Prod_grad/Reshapegradients/MC/Prod_grad/Rank"/device:CPU:0*
T0
{
gradients/MC/Prod_grad/mod_1FloorModgradients/MC/Prod_grad/add_1gradients/MC/Prod_grad/Rank"/device:CPU:0*
T0
]
$gradients/MC/Prod_grad/range_1/startConst"/device:CPU:0*
dtype0*
value	B : 
]
$gradients/MC/Prod_grad/range_1/deltaConst"/device:CPU:0*
value	B :*
dtype0
?
gradients/MC/Prod_grad/range_1Range$gradients/MC/Prod_grad/range_1/startgradients/MC/Prod_grad/Rank$gradients/MC/Prod_grad/range_1/delta"/device:CPU:0*

Tidx0
?
gradients/MC/Prod_grad/ListDiffListDiffgradients/MC/Prod_grad/range_1gradients/MC/Prod_grad/mod_1"/device:CPU:0*
T0*
out_idx0
[
"gradients/MC/Prod_grad/concat/axisConst"/device:CPU:0*
dtype0*
value	B : 
?
gradients/MC/Prod_grad/concatConcatV2gradients/MC/Prod_grad/mod_1gradients/MC/Prod_grad/ListDiff"gradients/MC/Prod_grad/concat/axis"/device:CPU:0*
N*

Tidx0*
T0
]
$gradients/MC/Prod_grad/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0
?
gradients/MC/Prod_grad/GatherV2GatherV2gradients/MC/Prod_grad/Shapegradients/MC/Prod_grad/mod_1$gradients/MC/Prod_grad/GatherV2/axis"/device:CPU:0*

batch_dims *
Tindices0*
Tparams0*
Taxis0
Y
gradients/MC/Prod_grad/ConstConst"/device:CPU:0*
dtype0*
valueB: 
?
gradients/MC/Prod_grad/ProdProdgradients/MC/Prod_grad/GatherV2gradients/MC/Prod_grad/Const"/device:CPU:0*

Tidx0*
	keep_dims( *
T0
_
&gradients/MC/Prod_grad/GatherV2_1/axisConst"/device:CPU:0*
value	B : *
dtype0
?
!gradients/MC/Prod_grad/GatherV2_1GatherV2gradients/MC/Prod_grad/Shapegradients/MC/Prod_grad/ListDiff&gradients/MC/Prod_grad/GatherV2_1/axis"/device:CPU:0*

batch_dims *
Tindices0*
Tparams0*
Taxis0
[
gradients/MC/Prod_grad/Const_1Const"/device:CPU:0*
dtype0*
valueB: 
?
gradients/MC/Prod_grad/Prod_1Prod!gradients/MC/Prod_grad/GatherV2_1gradients/MC/Prod_grad/Const_1"/device:CPU:0*
T0*

Tidx0*
	keep_dims( 
q
 gradients/MC/Prod_grad/transpose	TransposeMC/Reshape_11gradients/MC/Prod_grad/concat*
T0*
Tperm0
b
gradients/MC/Prod_grad/Shape_2Shape gradients/MC/Prod_grad/transpose*
T0*
out_type0
?
&gradients/MC/Prod_grad/Reshape_2/shapePackgradients/MC/Prod_grad/Prodgradients/MC/Prod_grad/Prod_1*
T0*

axis *
N
?
 gradients/MC/Prod_grad/Reshape_2Reshape gradients/MC/Prod_grad/transpose&gradients/MC/Prod_grad/Reshape_2/shape*
T0*
Tshape0
M
#gradients/MC/Prod_grad/Cumprod/axisConst*
value	B : *
dtype0
?
gradients/MC/Prod_grad/CumprodCumprod gradients/MC/Prod_grad/Reshape_2#gradients/MC/Prod_grad/Cumprod/axis*
reverse( *

Tidx0*
T0*
	exclusive(
O
%gradients/MC/Prod_grad/Cumprod_1/axisConst*
dtype0*
value	B : 
?
 gradients/MC/Prod_grad/Cumprod_1Cumprod gradients/MC/Prod_grad/Reshape_2%gradients/MC/Prod_grad/Cumprod_1/axis*
reverse(*

Tidx0*
T0*
	exclusive(
l
gradients/MC/Prod_grad/mulMulgradients/MC/Prod_grad/Cumprod gradients/MC/Prod_grad/Cumprod_1*
T0
~
 gradients/MC/Prod_grad/Reshape_3Reshapegradients/MC/Prod_grad/mulgradients/MC/Prod_grad/Shape_2*
T0*
Tshape0
e
(gradients/MC/Prod_grad/InvertPermutationInvertPermutationgradients/MC/Prod_grad/concat*
T0
?
"gradients/MC/Prod_grad/transpose_1	Transpose gradients/MC/Prod_grad/Reshape_3(gradients/MC/Prod_grad/InvertPermutation*
T0*
Tperm0
m
gradients/MC/Prod_grad/mul_1Mulgradients/MC/Prod_grad/Tile"gradients/MC/Prod_grad/transpose_1*
T0
~
 gradients/MC/Prod_grad/Reshape_4Reshapegradients/MC/Prod_grad/mul_1gradients/MC/Prod_grad/Shape*
T0*
Tshape0
W
"gradients/MC/Reshape_11_grad/ShapeShapeMC/deconv_6/add_1*
T0*
out_type0
?
$gradients/MC/Reshape_11_grad/ReshapeReshape gradients/MC/Prod_grad/Reshape_4"gradients/MC/Reshape_11_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_6/add_1_grad/ShapeShape$MC/deconv_6/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_6/add_1_grad/Shape_1ShapeMC/deconv_6/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_6/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_6/add_1_grad/Shape(gradients/MC/deconv_6/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_6/add_1_grad/SumSum$gradients/MC/Reshape_11_grad/Reshape6gradients/MC/deconv_6/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
?
(gradients/MC/deconv_6/add_1_grad/ReshapeReshape$gradients/MC/deconv_6/add_1_grad/Sum&gradients/MC/deconv_6/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_6/add_1_grad/Sum_1Sum$gradients/MC/Reshape_11_grad/Reshape8gradients/MC/deconv_6/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
*gradients/MC/deconv_6/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_6/add_1_grad/Sum_1(gradients/MC/deconv_6/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_6/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_6/add_1_grad/Reshape+^gradients/MC/deconv_6/add_1_grad/Reshape_1
?
9gradients/MC/deconv_6/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_6/add_1_grad/Reshape2^gradients/MC/deconv_6/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_6/add_1_grad/Reshape
?
;gradients/MC/deconv_6/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_6/add_1_grad/Reshape_12^gradients/MC/deconv_6/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_6/add_1_grad/Reshape_1
y
9gradients/MC/deconv_6/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_6/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_6/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_6/add_1_grad/tuple/control_dependency9gradients/MC/deconv_6/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_6/conv1d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
?
@gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_6/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_6/conv1d_transpose_grad/Shape'MC/deconv_6/conv1d_transpose/ExpandDims*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(
?
2gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_6/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_6/conv1d_transpose/ExpandDims_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
<gradients/MC/deconv_6/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_6/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_6/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_6/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_6/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_6/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_6/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_6/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_6/conv1d_transpose/ExpandDims_1_grad/ShapeConst*
dtype0*!
valueB"         
?
@gradients/MC/deconv_6/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_6/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_6/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_6/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_6/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_6/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_6/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_6/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_6/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_6/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_6/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_6/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_6/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_6/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_6/MaxPool1d/ExpandDimsMC/maxpool_6/MaxPool1d5gradients/MC/maxpool_6/MaxPool1d/Squeeze_grad/Reshape*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

g
6gradients/MC/maxpool_6/MaxPool1d/ExpandDims_grad/ShapeShapeMC/Reshape_10*
T0*
out_type0
?
8gradients/MC/maxpool_6/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_6/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_6/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
S
"gradients/MC/Reshape_10_grad/ShapeShapeMC/conv_6/add*
T0*
out_type0
?
$gradients/MC/Reshape_10_grad/ReshapeReshape8gradients/MC/maxpool_6/MaxPool1d/ExpandDims_grad/Reshape"gradients/MC/Reshape_10_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_6/add_grad/ShapeShapeMC/conv_6/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_6/add_grad/Shape_1ShapeMC/conv_6/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_6/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_6/add_grad/Shape$gradients/MC/conv_6/add_grad/Shape_1*
T0
?
 gradients/MC/conv_6/add_grad/SumSum$gradients/MC/Reshape_10_grad/Reshape2gradients/MC/conv_6/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
?
$gradients/MC/conv_6/add_grad/ReshapeReshape gradients/MC/conv_6/add_grad/Sum"gradients/MC/conv_6/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_6/add_grad/Sum_1Sum$gradients/MC/Reshape_10_grad/Reshape4gradients/MC/conv_6/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
&gradients/MC/conv_6/add_grad/Reshape_1Reshape"gradients/MC/conv_6/add_grad/Sum_1$gradients/MC/conv_6/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_6/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_6/add_grad/Reshape'^gradients/MC/conv_6/add_grad/Reshape_1
?
5gradients/MC/conv_6/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_6/add_grad/Reshape.^gradients/MC/conv_6/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_6/add_grad/Reshape
?
7gradients/MC/conv_6/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_6/add_grad/Reshape_1.^gradients/MC/conv_6/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_6/add_grad/Reshape_1
|
&gradients/MC/conv_6/Conv2D_grad/ShapeNShapeN
MC/Slice_4MC/conv_6/kernel/kernel/read*
T0*
out_type0*
N
?
3gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_6/Conv2D_grad/ShapeNMC/conv_6/kernel/kernel/read5gradients/MC/conv_6/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
4gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
MC/Slice_4(gradients/MC/conv_6/Conv2D_grad/ShapeN:15gradients/MC/conv_6/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
?
0gradients/MC/conv_6/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_6/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_6/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_6/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_6/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_6/Conv2D_grad/Conv2DBackpropFilter
H
gradients/MC/Slice_4_grad/RankConst*
dtype0*
value	B :
M
gradients/MC/Slice_4_grad/ShapeShape
MC/Slice_4*
T0*
out_type0
K
!gradients/MC/Slice_4_grad/stack/1Const*
value	B :*
dtype0
?
gradients/MC/Slice_4_grad/stackPackgradients/MC/Slice_4_grad/Rank!gradients/MC/Slice_4_grad/stack/1*
N*
T0*

axis 
v
!gradients/MC/Slice_4_grad/ReshapeReshapeMC/Slice_4/begingradients/MC/Slice_4_grad/stack*
T0*
Tshape0
N
!gradients/MC/Slice_4_grad/Shape_1Shape	MC/Tile_4*
T0*
out_type0
q
gradients/MC/Slice_4_grad/subSub!gradients/MC/Slice_4_grad/Shape_1gradients/MC/Slice_4_grad/Shape*
T0
`
gradients/MC/Slice_4_grad/sub_1Subgradients/MC/Slice_4_grad/subMC/Slice_4/begin*
T0
?
#gradients/MC/Slice_4_grad/Reshape_1Reshapegradients/MC/Slice_4_grad/sub_1gradients/MC/Slice_4_grad/stack*
T0*
Tshape0
O
%gradients/MC/Slice_4_grad/concat/axisConst*
dtype0*
value	B :
?
 gradients/MC/Slice_4_grad/concatConcatV2!gradients/MC/Slice_4_grad/Reshape#gradients/MC/Slice_4_grad/Reshape_1%gradients/MC/Slice_4_grad/concat/axis*
T0*
N*

Tidx0
?
gradients/MC/Slice_4_grad/PadPad8gradients/MC/conv_6/Conv2D_grad/tuple/control_dependency gradients/MC/Slice_4_grad/concat*
T0*
	Tpaddings0
N
gradients/MC/Tile_4_grad/ShapeShapeMC/Reshape_9*
T0*
out_type0
y
gradients/MC/Tile_4_grad/stackPackMC/Tile_4/multiplesgradients/MC/Tile_4_grad/Shape*
T0*

axis *
N
\
'gradients/MC/Tile_4_grad/transpose/permConst*
valueB"       *
dtype0
?
"gradients/MC/Tile_4_grad/transpose	Transposegradients/MC/Tile_4_grad/stack'gradients/MC/Tile_4_grad/transpose/perm*
T0*
Tperm0
]
&gradients/MC/Tile_4_grad/Reshape/shapeConst*
valueB:
?????????*
dtype0
?
 gradients/MC/Tile_4_grad/ReshapeReshape"gradients/MC/Tile_4_grad/transpose&gradients/MC/Tile_4_grad/Reshape/shape*
T0*
Tshape0
G
gradients/MC/Tile_4_grad/SizeConst*
value	B :*
dtype0
N
$gradients/MC/Tile_4_grad/range/startConst*
dtype0*
value	B : 
N
$gradients/MC/Tile_4_grad/range/deltaConst*
dtype0*
value	B :
?
gradients/MC/Tile_4_grad/rangeRange$gradients/MC/Tile_4_grad/range/startgradients/MC/Tile_4_grad/Size$gradients/MC/Tile_4_grad/range/delta*

Tidx0
?
"gradients/MC/Tile_4_grad/Reshape_1Reshapegradients/MC/Slice_4_grad/Pad gradients/MC/Tile_4_grad/Reshape*
T0*
Tshape0
?
gradients/MC/Tile_4_grad/SumSum"gradients/MC/Tile_4_grad/Reshape_1gradients/MC/Tile_4_grad/range*

Tidx0*
	keep_dims( *
T0
V
!gradients/MC/Reshape_9_grad/ShapeShapeMC/deconv_5/add_1*
T0*
out_type0
?
#gradients/MC/Reshape_9_grad/ReshapeReshapegradients/MC/Tile_4_grad/Sum!gradients/MC/Reshape_9_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_5/add_1_grad/ShapeShape$MC/deconv_5/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_5/add_1_grad/Shape_1ShapeMC/deconv_5/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_5/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_5/add_1_grad/Shape(gradients/MC/deconv_5/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_5/add_1_grad/SumSum#gradients/MC/Reshape_9_grad/Reshape6gradients/MC/deconv_5/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
(gradients/MC/deconv_5/add_1_grad/ReshapeReshape$gradients/MC/deconv_5/add_1_grad/Sum&gradients/MC/deconv_5/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_5/add_1_grad/Sum_1Sum#gradients/MC/Reshape_9_grad/Reshape8gradients/MC/deconv_5/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
*gradients/MC/deconv_5/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_5/add_1_grad/Sum_1(gradients/MC/deconv_5/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_5/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_5/add_1_grad/Reshape+^gradients/MC/deconv_5/add_1_grad/Reshape_1
?
9gradients/MC/deconv_5/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_5/add_1_grad/Reshape2^gradients/MC/deconv_5/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_5/add_1_grad/Reshape
?
;gradients/MC/deconv_5/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_5/add_1_grad/Reshape_12^gradients/MC/deconv_5/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_5/add_1_grad/Reshape_1
y
9gradients/MC/deconv_5/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_5/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_5/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_5/add_1_grad/tuple/control_dependency9gradients/MC/deconv_5/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_5/conv1d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
?
@gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_5/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_5/conv1d_transpose_grad/Shape'MC/deconv_5/conv1d_transpose/ExpandDims*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
	dilations
*
T0
?
2gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_5/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_5/conv1d_transpose/ExpandDims_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
<gradients/MC/deconv_5/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_5/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_5/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_5/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_5/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_5/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_5/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_5/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_5/conv1d_transpose/ExpandDims_1_grad/ShapeConst*
dtype0*!
valueB"         
?
@gradients/MC/deconv_5/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_5/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_5/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_5/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_5/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_5/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_5/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_5/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_5/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_5/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_5/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_5/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_5/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_5/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_5/MaxPool1d/ExpandDimsMC/maxpool_5/MaxPool1d5gradients/MC/maxpool_5/MaxPool1d/Squeeze_grad/Reshape*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

f
6gradients/MC/maxpool_5/MaxPool1d/ExpandDims_grad/ShapeShapeMC/Reshape_8*
T0*
out_type0
?
8gradients/MC/maxpool_5/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_5/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_5/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
R
!gradients/MC/Reshape_8_grad/ShapeShapeMC/conv_5/add*
T0*
out_type0
?
#gradients/MC/Reshape_8_grad/ReshapeReshape8gradients/MC/maxpool_5/MaxPool1d/ExpandDims_grad/Reshape!gradients/MC/Reshape_8_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_5/add_grad/ShapeShapeMC/conv_5/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_5/add_grad/Shape_1ShapeMC/conv_5/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_5/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_5/add_grad/Shape$gradients/MC/conv_5/add_grad/Shape_1*
T0
?
 gradients/MC/conv_5/add_grad/SumSum#gradients/MC/Reshape_8_grad/Reshape2gradients/MC/conv_5/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
$gradients/MC/conv_5/add_grad/ReshapeReshape gradients/MC/conv_5/add_grad/Sum"gradients/MC/conv_5/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_5/add_grad/Sum_1Sum#gradients/MC/Reshape_8_grad/Reshape4gradients/MC/conv_5/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
?
&gradients/MC/conv_5/add_grad/Reshape_1Reshape"gradients/MC/conv_5/add_grad/Sum_1$gradients/MC/conv_5/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_5/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_5/add_grad/Reshape'^gradients/MC/conv_5/add_grad/Reshape_1
?
5gradients/MC/conv_5/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_5/add_grad/Reshape.^gradients/MC/conv_5/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_5/add_grad/Reshape
?
7gradients/MC/conv_5/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_5/add_grad/Reshape_1.^gradients/MC/conv_5/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_5/add_grad/Reshape_1
|
&gradients/MC/conv_5/Conv2D_grad/ShapeNShapeN
MC/Slice_3MC/conv_5/kernel/kernel/read*
N*
T0*
out_type0
?
3gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_5/Conv2D_grad/ShapeNMC/conv_5/kernel/kernel/read5gradients/MC/conv_5/add_grad/tuple/control_dependency*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
?
4gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
MC/Slice_3(gradients/MC/conv_5/Conv2D_grad/ShapeN:15gradients/MC/conv_5/add_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
?
0gradients/MC/conv_5/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_5/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_5/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_5/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_5/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_5/Conv2D_grad/Conv2DBackpropFilter
H
gradients/MC/Slice_3_grad/RankConst*
value	B :*
dtype0
M
gradients/MC/Slice_3_grad/ShapeShape
MC/Slice_3*
T0*
out_type0
K
!gradients/MC/Slice_3_grad/stack/1Const*
value	B :*
dtype0
?
gradients/MC/Slice_3_grad/stackPackgradients/MC/Slice_3_grad/Rank!gradients/MC/Slice_3_grad/stack/1*
T0*

axis *
N
v
!gradients/MC/Slice_3_grad/ReshapeReshapeMC/Slice_3/begingradients/MC/Slice_3_grad/stack*
T0*
Tshape0
N
!gradients/MC/Slice_3_grad/Shape_1Shape	MC/Tile_3*
T0*
out_type0
q
gradients/MC/Slice_3_grad/subSub!gradients/MC/Slice_3_grad/Shape_1gradients/MC/Slice_3_grad/Shape*
T0
`
gradients/MC/Slice_3_grad/sub_1Subgradients/MC/Slice_3_grad/subMC/Slice_3/begin*
T0
?
#gradients/MC/Slice_3_grad/Reshape_1Reshapegradients/MC/Slice_3_grad/sub_1gradients/MC/Slice_3_grad/stack*
T0*
Tshape0
O
%gradients/MC/Slice_3_grad/concat/axisConst*
value	B :*
dtype0
?
 gradients/MC/Slice_3_grad/concatConcatV2!gradients/MC/Slice_3_grad/Reshape#gradients/MC/Slice_3_grad/Reshape_1%gradients/MC/Slice_3_grad/concat/axis*
T0*
N*

Tidx0
?
gradients/MC/Slice_3_grad/PadPad8gradients/MC/conv_5/Conv2D_grad/tuple/control_dependency gradients/MC/Slice_3_grad/concat*
	Tpaddings0*
T0
N
gradients/MC/Tile_3_grad/ShapeShapeMC/Reshape_7*
T0*
out_type0
y
gradients/MC/Tile_3_grad/stackPackMC/Tile_3/multiplesgradients/MC/Tile_3_grad/Shape*
N*
T0*

axis 
\
'gradients/MC/Tile_3_grad/transpose/permConst*
valueB"       *
dtype0
?
"gradients/MC/Tile_3_grad/transpose	Transposegradients/MC/Tile_3_grad/stack'gradients/MC/Tile_3_grad/transpose/perm*
Tperm0*
T0
]
&gradients/MC/Tile_3_grad/Reshape/shapeConst*
valueB:
?????????*
dtype0
?
 gradients/MC/Tile_3_grad/ReshapeReshape"gradients/MC/Tile_3_grad/transpose&gradients/MC/Tile_3_grad/Reshape/shape*
T0*
Tshape0
G
gradients/MC/Tile_3_grad/SizeConst*
dtype0*
value	B :
N
$gradients/MC/Tile_3_grad/range/startConst*
value	B : *
dtype0
N
$gradients/MC/Tile_3_grad/range/deltaConst*
value	B :*
dtype0
?
gradients/MC/Tile_3_grad/rangeRange$gradients/MC/Tile_3_grad/range/startgradients/MC/Tile_3_grad/Size$gradients/MC/Tile_3_grad/range/delta*

Tidx0
?
"gradients/MC/Tile_3_grad/Reshape_1Reshapegradients/MC/Slice_3_grad/Pad gradients/MC/Tile_3_grad/Reshape*
T0*
Tshape0
?
gradients/MC/Tile_3_grad/SumSum"gradients/MC/Tile_3_grad/Reshape_1gradients/MC/Tile_3_grad/range*

Tidx0*
	keep_dims( *
T0
V
!gradients/MC/Reshape_7_grad/ShapeShapeMC/deconv_4/add_1*
T0*
out_type0
?
#gradients/MC/Reshape_7_grad/ReshapeReshapegradients/MC/Tile_3_grad/Sum!gradients/MC/Reshape_7_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_4/add_1_grad/ShapeShape$MC/deconv_4/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_4/add_1_grad/Shape_1ShapeMC/deconv_4/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_4/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_4/add_1_grad/Shape(gradients/MC/deconv_4/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_4/add_1_grad/SumSum#gradients/MC/Reshape_7_grad/Reshape6gradients/MC/deconv_4/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
?
(gradients/MC/deconv_4/add_1_grad/ReshapeReshape$gradients/MC/deconv_4/add_1_grad/Sum&gradients/MC/deconv_4/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_4/add_1_grad/Sum_1Sum#gradients/MC/Reshape_7_grad/Reshape8gradients/MC/deconv_4/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
*gradients/MC/deconv_4/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_4/add_1_grad/Sum_1(gradients/MC/deconv_4/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_4/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_4/add_1_grad/Reshape+^gradients/MC/deconv_4/add_1_grad/Reshape_1
?
9gradients/MC/deconv_4/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_4/add_1_grad/Reshape2^gradients/MC/deconv_4/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_4/add_1_grad/Reshape
?
;gradients/MC/deconv_4/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_4/add_1_grad/Reshape_12^gradients/MC/deconv_4/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_4/add_1_grad/Reshape_1
y
9gradients/MC/deconv_4/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_4/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_4/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_4/add_1_grad/tuple/control_dependency9gradients/MC/deconv_4/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_4/conv1d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
?
@gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_4/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_4/conv1d_transpose_grad/Shape'MC/deconv_4/conv1d_transpose/ExpandDims*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
2gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_4/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_4/conv1d_transpose/ExpandDims_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
<gradients/MC/deconv_4/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_4/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_4/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_4/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_4/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_4/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_4/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_4/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_4/conv1d_transpose/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0
?
@gradients/MC/deconv_4/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_4/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_4/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_4/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_4/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_4/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_4/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_4/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_4/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_4/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_4/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_4/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_4/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_4/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_4/MaxPool1d/ExpandDimsMC/maxpool_4/MaxPool1d5gradients/MC/maxpool_4/MaxPool1d/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
f
6gradients/MC/maxpool_4/MaxPool1d/ExpandDims_grad/ShapeShapeMC/Reshape_6*
T0*
out_type0
?
8gradients/MC/maxpool_4/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_4/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_4/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
R
!gradients/MC/Reshape_6_grad/ShapeShapeMC/conv_4/add*
T0*
out_type0
?
#gradients/MC/Reshape_6_grad/ReshapeReshape8gradients/MC/maxpool_4/MaxPool1d/ExpandDims_grad/Reshape!gradients/MC/Reshape_6_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_4/add_grad/ShapeShapeMC/conv_4/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_4/add_grad/Shape_1ShapeMC/conv_4/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_4/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_4/add_grad/Shape$gradients/MC/conv_4/add_grad/Shape_1*
T0
?
 gradients/MC/conv_4/add_grad/SumSum#gradients/MC/Reshape_6_grad/Reshape2gradients/MC/conv_4/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
?
$gradients/MC/conv_4/add_grad/ReshapeReshape gradients/MC/conv_4/add_grad/Sum"gradients/MC/conv_4/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_4/add_grad/Sum_1Sum#gradients/MC/Reshape_6_grad/Reshape4gradients/MC/conv_4/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
?
&gradients/MC/conv_4/add_grad/Reshape_1Reshape"gradients/MC/conv_4/add_grad/Sum_1$gradients/MC/conv_4/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_4/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_4/add_grad/Reshape'^gradients/MC/conv_4/add_grad/Reshape_1
?
5gradients/MC/conv_4/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_4/add_grad/Reshape.^gradients/MC/conv_4/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_4/add_grad/Reshape
?
7gradients/MC/conv_4/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_4/add_grad/Reshape_1.^gradients/MC/conv_4/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_4/add_grad/Reshape_1
|
&gradients/MC/conv_4/Conv2D_grad/ShapeNShapeN
MC/Slice_2MC/conv_4/kernel/kernel/read*
T0*
out_type0*
N
?
3gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_4/Conv2D_grad/ShapeNMC/conv_4/kernel/kernel/read5gradients/MC/conv_4/add_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0
?
4gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
MC/Slice_2(gradients/MC/conv_4/Conv2D_grad/ShapeN:15gradients/MC/conv_4/add_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
0gradients/MC/conv_4/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_4/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_4/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_4/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_4/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_4/Conv2D_grad/Conv2DBackpropFilter
H
gradients/MC/Slice_2_grad/RankConst*
value	B :*
dtype0
M
gradients/MC/Slice_2_grad/ShapeShape
MC/Slice_2*
T0*
out_type0
K
!gradients/MC/Slice_2_grad/stack/1Const*
value	B :*
dtype0
?
gradients/MC/Slice_2_grad/stackPackgradients/MC/Slice_2_grad/Rank!gradients/MC/Slice_2_grad/stack/1*
T0*

axis *
N
v
!gradients/MC/Slice_2_grad/ReshapeReshapeMC/Slice_2/begingradients/MC/Slice_2_grad/stack*
T0*
Tshape0
N
!gradients/MC/Slice_2_grad/Shape_1Shape	MC/Tile_2*
T0*
out_type0
q
gradients/MC/Slice_2_grad/subSub!gradients/MC/Slice_2_grad/Shape_1gradients/MC/Slice_2_grad/Shape*
T0
`
gradients/MC/Slice_2_grad/sub_1Subgradients/MC/Slice_2_grad/subMC/Slice_2/begin*
T0
?
#gradients/MC/Slice_2_grad/Reshape_1Reshapegradients/MC/Slice_2_grad/sub_1gradients/MC/Slice_2_grad/stack*
T0*
Tshape0
O
%gradients/MC/Slice_2_grad/concat/axisConst*
value	B :*
dtype0
?
 gradients/MC/Slice_2_grad/concatConcatV2!gradients/MC/Slice_2_grad/Reshape#gradients/MC/Slice_2_grad/Reshape_1%gradients/MC/Slice_2_grad/concat/axis*

Tidx0*
T0*
N
?
gradients/MC/Slice_2_grad/PadPad8gradients/MC/conv_4/Conv2D_grad/tuple/control_dependency gradients/MC/Slice_2_grad/concat*
T0*
	Tpaddings0
N
gradients/MC/Tile_2_grad/ShapeShapeMC/Reshape_5*
T0*
out_type0
y
gradients/MC/Tile_2_grad/stackPackMC/Tile_2/multiplesgradients/MC/Tile_2_grad/Shape*
T0*

axis *
N
\
'gradients/MC/Tile_2_grad/transpose/permConst*
dtype0*
valueB"       
?
"gradients/MC/Tile_2_grad/transpose	Transposegradients/MC/Tile_2_grad/stack'gradients/MC/Tile_2_grad/transpose/perm*
Tperm0*
T0
]
&gradients/MC/Tile_2_grad/Reshape/shapeConst*
valueB:
?????????*
dtype0
?
 gradients/MC/Tile_2_grad/ReshapeReshape"gradients/MC/Tile_2_grad/transpose&gradients/MC/Tile_2_grad/Reshape/shape*
T0*
Tshape0
G
gradients/MC/Tile_2_grad/SizeConst*
dtype0*
value	B :
N
$gradients/MC/Tile_2_grad/range/startConst*
value	B : *
dtype0
N
$gradients/MC/Tile_2_grad/range/deltaConst*
value	B :*
dtype0
?
gradients/MC/Tile_2_grad/rangeRange$gradients/MC/Tile_2_grad/range/startgradients/MC/Tile_2_grad/Size$gradients/MC/Tile_2_grad/range/delta*

Tidx0
?
"gradients/MC/Tile_2_grad/Reshape_1Reshapegradients/MC/Slice_2_grad/Pad gradients/MC/Tile_2_grad/Reshape*
T0*
Tshape0
?
gradients/MC/Tile_2_grad/SumSum"gradients/MC/Tile_2_grad/Reshape_1gradients/MC/Tile_2_grad/range*

Tidx0*
	keep_dims( *
T0
V
!gradients/MC/Reshape_5_grad/ShapeShapeMC/deconv_3/add_1*
T0*
out_type0
?
#gradients/MC/Reshape_5_grad/ReshapeReshapegradients/MC/Tile_2_grad/Sum!gradients/MC/Reshape_5_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_3/add_1_grad/ShapeShape$MC/deconv_3/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_3/add_1_grad/Shape_1ShapeMC/deconv_3/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_3/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_3/add_1_grad/Shape(gradients/MC/deconv_3/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_3/add_1_grad/SumSum#gradients/MC/Reshape_5_grad/Reshape6gradients/MC/deconv_3/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
(gradients/MC/deconv_3/add_1_grad/ReshapeReshape$gradients/MC/deconv_3/add_1_grad/Sum&gradients/MC/deconv_3/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_3/add_1_grad/Sum_1Sum#gradients/MC/Reshape_5_grad/Reshape8gradients/MC/deconv_3/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
*gradients/MC/deconv_3/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_3/add_1_grad/Sum_1(gradients/MC/deconv_3/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_3/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_3/add_1_grad/Reshape+^gradients/MC/deconv_3/add_1_grad/Reshape_1
?
9gradients/MC/deconv_3/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_3/add_1_grad/Reshape2^gradients/MC/deconv_3/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_3/add_1_grad/Reshape
?
;gradients/MC/deconv_3/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_3/add_1_grad/Reshape_12^gradients/MC/deconv_3/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_3/add_1_grad/Reshape_1
y
9gradients/MC/deconv_3/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_3/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_3/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_3/add_1_grad/tuple/control_dependency9gradients/MC/deconv_3/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_3/conv1d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
?
@gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_3/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_3/conv1d_transpose_grad/Shape'MC/deconv_3/conv1d_transpose/ExpandDims*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
2gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_3/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_3/conv1d_transpose/ExpandDims_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
?
<gradients/MC/deconv_3/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_3/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_3/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_3/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_3/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_3/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_3/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_3/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_3/conv1d_transpose/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0
?
@gradients/MC/deconv_3/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_3/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_3/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_3/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_3/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_3/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_3/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_3/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_3/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_3/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_3/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_3/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_3/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_3/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_3/MaxPool1d/ExpandDimsMC/maxpool_3/MaxPool1d5gradients/MC/maxpool_3/MaxPool1d/Squeeze_grad/Reshape*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

f
6gradients/MC/maxpool_3/MaxPool1d/ExpandDims_grad/ShapeShapeMC/Reshape_4*
T0*
out_type0
?
8gradients/MC/maxpool_3/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_3/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_3/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
R
!gradients/MC/Reshape_4_grad/ShapeShapeMC/conv_3/add*
T0*
out_type0
?
#gradients/MC/Reshape_4_grad/ReshapeReshape8gradients/MC/maxpool_3/MaxPool1d/ExpandDims_grad/Reshape!gradients/MC/Reshape_4_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_3/add_grad/ShapeShapeMC/conv_3/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_3/add_grad/Shape_1ShapeMC/conv_3/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_3/add_grad/Shape$gradients/MC/conv_3/add_grad/Shape_1*
T0
?
 gradients/MC/conv_3/add_grad/SumSum#gradients/MC/Reshape_4_grad/Reshape2gradients/MC/conv_3/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
$gradients/MC/conv_3/add_grad/ReshapeReshape gradients/MC/conv_3/add_grad/Sum"gradients/MC/conv_3/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_3/add_grad/Sum_1Sum#gradients/MC/Reshape_4_grad/Reshape4gradients/MC/conv_3/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
&gradients/MC/conv_3/add_grad/Reshape_1Reshape"gradients/MC/conv_3/add_grad/Sum_1$gradients/MC/conv_3/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_3/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_3/add_grad/Reshape'^gradients/MC/conv_3/add_grad/Reshape_1
?
5gradients/MC/conv_3/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_3/add_grad/Reshape.^gradients/MC/conv_3/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_3/add_grad/Reshape
?
7gradients/MC/conv_3/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_3/add_grad/Reshape_1.^gradients/MC/conv_3/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_3/add_grad/Reshape_1
|
&gradients/MC/conv_3/Conv2D_grad/ShapeNShapeN
MC/Slice_1MC/conv_3/kernel/kernel/read*
T0*
out_type0*
N
?
3gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_3/Conv2D_grad/ShapeNMC/conv_3/kernel/kernel/read5gradients/MC/conv_3/add_grad/tuple/control_dependency*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 
?
4gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter
MC/Slice_1(gradients/MC/conv_3/Conv2D_grad/ShapeN:15gradients/MC/conv_3/add_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
0gradients/MC/conv_3/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_3/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_3/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_3/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_3/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_3/Conv2D_grad/Conv2DBackpropFilter
H
gradients/MC/Slice_1_grad/RankConst*
value	B :*
dtype0
M
gradients/MC/Slice_1_grad/ShapeShape
MC/Slice_1*
T0*
out_type0
K
!gradients/MC/Slice_1_grad/stack/1Const*
dtype0*
value	B :
?
gradients/MC/Slice_1_grad/stackPackgradients/MC/Slice_1_grad/Rank!gradients/MC/Slice_1_grad/stack/1*
N*
T0*

axis 
v
!gradients/MC/Slice_1_grad/ReshapeReshapeMC/Slice_1/begingradients/MC/Slice_1_grad/stack*
T0*
Tshape0
N
!gradients/MC/Slice_1_grad/Shape_1Shape	MC/Tile_1*
T0*
out_type0
q
gradients/MC/Slice_1_grad/subSub!gradients/MC/Slice_1_grad/Shape_1gradients/MC/Slice_1_grad/Shape*
T0
`
gradients/MC/Slice_1_grad/sub_1Subgradients/MC/Slice_1_grad/subMC/Slice_1/begin*
T0
?
#gradients/MC/Slice_1_grad/Reshape_1Reshapegradients/MC/Slice_1_grad/sub_1gradients/MC/Slice_1_grad/stack*
T0*
Tshape0
O
%gradients/MC/Slice_1_grad/concat/axisConst*
value	B :*
dtype0
?
 gradients/MC/Slice_1_grad/concatConcatV2!gradients/MC/Slice_1_grad/Reshape#gradients/MC/Slice_1_grad/Reshape_1%gradients/MC/Slice_1_grad/concat/axis*

Tidx0*
T0*
N
?
gradients/MC/Slice_1_grad/PadPad8gradients/MC/conv_3/Conv2D_grad/tuple/control_dependency gradients/MC/Slice_1_grad/concat*
T0*
	Tpaddings0
N
gradients/MC/Tile_1_grad/ShapeShapeMC/Reshape_3*
T0*
out_type0
y
gradients/MC/Tile_1_grad/stackPackMC/Tile_1/multiplesgradients/MC/Tile_1_grad/Shape*
T0*

axis *
N
\
'gradients/MC/Tile_1_grad/transpose/permConst*
valueB"       *
dtype0
?
"gradients/MC/Tile_1_grad/transpose	Transposegradients/MC/Tile_1_grad/stack'gradients/MC/Tile_1_grad/transpose/perm*
Tperm0*
T0
]
&gradients/MC/Tile_1_grad/Reshape/shapeConst*
dtype0*
valueB:
?????????
?
 gradients/MC/Tile_1_grad/ReshapeReshape"gradients/MC/Tile_1_grad/transpose&gradients/MC/Tile_1_grad/Reshape/shape*
T0*
Tshape0
G
gradients/MC/Tile_1_grad/SizeConst*
value	B :*
dtype0
N
$gradients/MC/Tile_1_grad/range/startConst*
value	B : *
dtype0
N
$gradients/MC/Tile_1_grad/range/deltaConst*
value	B :*
dtype0
?
gradients/MC/Tile_1_grad/rangeRange$gradients/MC/Tile_1_grad/range/startgradients/MC/Tile_1_grad/Size$gradients/MC/Tile_1_grad/range/delta*

Tidx0
?
"gradients/MC/Tile_1_grad/Reshape_1Reshapegradients/MC/Slice_1_grad/Pad gradients/MC/Tile_1_grad/Reshape*
T0*
Tshape0
?
gradients/MC/Tile_1_grad/SumSum"gradients/MC/Tile_1_grad/Reshape_1gradients/MC/Tile_1_grad/range*

Tidx0*
	keep_dims( *
T0
V
!gradients/MC/Reshape_3_grad/ShapeShapeMC/deconv_2/add_1*
T0*
out_type0
?
#gradients/MC/Reshape_3_grad/ReshapeReshapegradients/MC/Tile_1_grad/Sum!gradients/MC/Reshape_3_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_2/add_1_grad/ShapeShape$MC/deconv_2/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_2/add_1_grad/Shape_1ShapeMC/deconv_2/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_2/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_2/add_1_grad/Shape(gradients/MC/deconv_2/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_2/add_1_grad/SumSum#gradients/MC/Reshape_3_grad/Reshape6gradients/MC/deconv_2/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
(gradients/MC/deconv_2/add_1_grad/ReshapeReshape$gradients/MC/deconv_2/add_1_grad/Sum&gradients/MC/deconv_2/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_2/add_1_grad/Sum_1Sum#gradients/MC/Reshape_3_grad/Reshape8gradients/MC/deconv_2/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
?
*gradients/MC/deconv_2/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_2/add_1_grad/Sum_1(gradients/MC/deconv_2/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_2/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_2/add_1_grad/Reshape+^gradients/MC/deconv_2/add_1_grad/Reshape_1
?
9gradients/MC/deconv_2/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_2/add_1_grad/Reshape2^gradients/MC/deconv_2/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_2/add_1_grad/Reshape
?
;gradients/MC/deconv_2/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_2/add_1_grad/Reshape_12^gradients/MC/deconv_2/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_2/add_1_grad/Reshape_1
y
9gradients/MC/deconv_2/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_2/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_2/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_2/add_1_grad/tuple/control_dependency9gradients/MC/deconv_2/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_2/conv1d_transpose_grad/ShapeConst*%
valueB"            *
dtype0
?
@gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_2/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_2/conv1d_transpose_grad/Shape'MC/deconv_2/conv1d_transpose/ExpandDims*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
2gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_2/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_2/conv1d_transpose/ExpandDims_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
	dilations

?
<gradients/MC/deconv_2/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_2/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_2/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_2/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_2/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_2/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_2/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_2/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_2/conv1d_transpose/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0
?
@gradients/MC/deconv_2/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_2/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_2/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_2/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_2/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_2/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_2/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_2/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_2/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_2/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_2/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_2/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_2/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_2/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_2/MaxPool1d/ExpandDimsMC/maxpool_2/MaxPool1d5gradients/MC/maxpool_2/MaxPool1d/Squeeze_grad/Reshape*
ksize
*
paddingVALID*
T0*
data_formatNHWC*
strides

f
6gradients/MC/maxpool_2/MaxPool1d/ExpandDims_grad/ShapeShapeMC/Reshape_2*
T0*
out_type0
?
8gradients/MC/maxpool_2/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_2/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_2/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
R
!gradients/MC/Reshape_2_grad/ShapeShapeMC/conv_2/add*
T0*
out_type0
?
#gradients/MC/Reshape_2_grad/ReshapeReshape8gradients/MC/maxpool_2/MaxPool1d/ExpandDims_grad/Reshape!gradients/MC/Reshape_2_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_2/add_grad/ShapeShapeMC/conv_2/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_2/add_grad/Shape_1ShapeMC/conv_2/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_2/add_grad/Shape$gradients/MC/conv_2/add_grad/Shape_1*
T0
?
 gradients/MC/conv_2/add_grad/SumSum#gradients/MC/Reshape_2_grad/Reshape2gradients/MC/conv_2/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
$gradients/MC/conv_2/add_grad/ReshapeReshape gradients/MC/conv_2/add_grad/Sum"gradients/MC/conv_2/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_2/add_grad/Sum_1Sum#gradients/MC/Reshape_2_grad/Reshape4gradients/MC/conv_2/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
&gradients/MC/conv_2/add_grad/Reshape_1Reshape"gradients/MC/conv_2/add_grad/Sum_1$gradients/MC/conv_2/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_2/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_2/add_grad/Reshape'^gradients/MC/conv_2/add_grad/Reshape_1
?
5gradients/MC/conv_2/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_2/add_grad/Reshape.^gradients/MC/conv_2/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_2/add_grad/Reshape
?
7gradients/MC/conv_2/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_2/add_grad/Reshape_1.^gradients/MC/conv_2/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_2/add_grad/Reshape_1
z
&gradients/MC/conv_2/Conv2D_grad/ShapeNShapeNMC/SliceMC/conv_2/kernel/kernel/read*
T0*
out_type0*
N
?
3gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_2/Conv2D_grad/ShapeNMC/conv_2/kernel/kernel/read5gradients/MC/conv_2/add_grad/tuple/control_dependency*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 
?
4gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterMC/Slice(gradients/MC/conv_2/Conv2D_grad/ShapeN:15gradients/MC/conv_2/add_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0
?
0gradients/MC/conv_2/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_2/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_2/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_2/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_2/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_2/Conv2D_grad/Conv2DBackpropFilter
F
gradients/MC/Slice_grad/RankConst*
value	B :*
dtype0
I
gradients/MC/Slice_grad/ShapeShapeMC/Slice*
T0*
out_type0
I
gradients/MC/Slice_grad/stack/1Const*
value	B :*
dtype0
?
gradients/MC/Slice_grad/stackPackgradients/MC/Slice_grad/Rankgradients/MC/Slice_grad/stack/1*
T0*

axis *
N
p
gradients/MC/Slice_grad/ReshapeReshapeMC/Slice/begingradients/MC/Slice_grad/stack*
T0*
Tshape0
J
gradients/MC/Slice_grad/Shape_1ShapeMC/Tile*
T0*
out_type0
k
gradients/MC/Slice_grad/subSubgradients/MC/Slice_grad/Shape_1gradients/MC/Slice_grad/Shape*
T0
Z
gradients/MC/Slice_grad/sub_1Subgradients/MC/Slice_grad/subMC/Slice/begin*
T0
?
!gradients/MC/Slice_grad/Reshape_1Reshapegradients/MC/Slice_grad/sub_1gradients/MC/Slice_grad/stack*
T0*
Tshape0
M
#gradients/MC/Slice_grad/concat/axisConst*
dtype0*
value	B :
?
gradients/MC/Slice_grad/concatConcatV2gradients/MC/Slice_grad/Reshape!gradients/MC/Slice_grad/Reshape_1#gradients/MC/Slice_grad/concat/axis*

Tidx0*
T0*
N
?
gradients/MC/Slice_grad/PadPad8gradients/MC/conv_2/Conv2D_grad/tuple/control_dependencygradients/MC/Slice_grad/concat*
	Tpaddings0*
T0
L
gradients/MC/Tile_grad/ShapeShapeMC/Reshape_1*
T0*
out_type0
s
gradients/MC/Tile_grad/stackPackMC/Tile/multiplesgradients/MC/Tile_grad/Shape*
T0*

axis *
N
Z
%gradients/MC/Tile_grad/transpose/permConst*
valueB"       *
dtype0
?
 gradients/MC/Tile_grad/transpose	Transposegradients/MC/Tile_grad/stack%gradients/MC/Tile_grad/transpose/perm*
T0*
Tperm0
[
$gradients/MC/Tile_grad/Reshape/shapeConst*
valueB:
?????????*
dtype0
?
gradients/MC/Tile_grad/ReshapeReshape gradients/MC/Tile_grad/transpose$gradients/MC/Tile_grad/Reshape/shape*
T0*
Tshape0
E
gradients/MC/Tile_grad/SizeConst*
value	B :*
dtype0
L
"gradients/MC/Tile_grad/range/startConst*
dtype0*
value	B : 
L
"gradients/MC/Tile_grad/range/deltaConst*
value	B :*
dtype0
?
gradients/MC/Tile_grad/rangeRange"gradients/MC/Tile_grad/range/startgradients/MC/Tile_grad/Size"gradients/MC/Tile_grad/range/delta*

Tidx0

 gradients/MC/Tile_grad/Reshape_1Reshapegradients/MC/Slice_grad/Padgradients/MC/Tile_grad/Reshape*
T0*
Tshape0
?
gradients/MC/Tile_grad/SumSum gradients/MC/Tile_grad/Reshape_1gradients/MC/Tile_grad/range*
T0*

Tidx0*
	keep_dims( 
V
!gradients/MC/Reshape_1_grad/ShapeShapeMC/deconv_1/add_1*
T0*
out_type0
?
#gradients/MC/Reshape_1_grad/ReshapeReshapegradients/MC/Tile_grad/Sum!gradients/MC/Reshape_1_grad/Shape*
T0*
Tshape0
n
&gradients/MC/deconv_1/add_1_grad/ShapeShape$MC/deconv_1/conv1d_transpose/Squeeze*
T0*
out_type0
f
(gradients/MC/deconv_1/add_1_grad/Shape_1ShapeMC/deconv_1/bias/bias/read*
T0*
out_type0
?
6gradients/MC/deconv_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/MC/deconv_1/add_1_grad/Shape(gradients/MC/deconv_1/add_1_grad/Shape_1*
T0
?
$gradients/MC/deconv_1/add_1_grad/SumSum#gradients/MC/Reshape_1_grad/Reshape6gradients/MC/deconv_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
?
(gradients/MC/deconv_1/add_1_grad/ReshapeReshape$gradients/MC/deconv_1/add_1_grad/Sum&gradients/MC/deconv_1/add_1_grad/Shape*
T0*
Tshape0
?
&gradients/MC/deconv_1/add_1_grad/Sum_1Sum#gradients/MC/Reshape_1_grad/Reshape8gradients/MC/deconv_1/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
?
*gradients/MC/deconv_1/add_1_grad/Reshape_1Reshape&gradients/MC/deconv_1/add_1_grad/Sum_1(gradients/MC/deconv_1/add_1_grad/Shape_1*
T0*
Tshape0
?
1gradients/MC/deconv_1/add_1_grad/tuple/group_depsNoOp)^gradients/MC/deconv_1/add_1_grad/Reshape+^gradients/MC/deconv_1/add_1_grad/Reshape_1
?
9gradients/MC/deconv_1/add_1_grad/tuple/control_dependencyIdentity(gradients/MC/deconv_1/add_1_grad/Reshape2^gradients/MC/deconv_1/add_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/MC/deconv_1/add_1_grad/Reshape
?
;gradients/MC/deconv_1/add_1_grad/tuple/control_dependency_1Identity*gradients/MC/deconv_1/add_1_grad/Reshape_12^gradients/MC/deconv_1/add_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/MC/deconv_1/add_1_grad/Reshape_1
y
9gradients/MC/deconv_1/conv1d_transpose/Squeeze_grad/ShapeShapeMC/deconv_1/conv1d_transpose*
T0*
out_type0
?
;gradients/MC/deconv_1/conv1d_transpose/Squeeze_grad/ReshapeReshape9gradients/MC/deconv_1/add_1_grad/tuple/control_dependency9gradients/MC/deconv_1/conv1d_transpose/Squeeze_grad/Shape*
T0*
Tshape0
n
1gradients/MC/deconv_1/conv1d_transpose_grad/ShapeConst*
dtype0*%
valueB"             
?
@gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DBackpropFilterConv2DBackpropFilter;gradients/MC/deconv_1/conv1d_transpose/Squeeze_grad/Reshape1gradients/MC/deconv_1/conv1d_transpose_grad/Shape'MC/deconv_1/conv1d_transpose/ExpandDims*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
2gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DConv2D;gradients/MC/deconv_1/conv1d_transpose/Squeeze_grad/Reshape)MC/deconv_1/conv1d_transpose/ExpandDims_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
<gradients/MC/deconv_1/conv1d_transpose_grad/tuple/group_depsNoOp3^gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DA^gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DBackpropFilter
?
Dgradients/MC/deconv_1/conv1d_transpose_grad/tuple/control_dependencyIdentity@gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DBackpropFilter=^gradients/MC/deconv_1/conv1d_transpose_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/MC/deconv_1/conv1d_transpose_grad/Conv2DBackpropFilter
?
Fgradients/MC/deconv_1/conv1d_transpose_grad/tuple/control_dependency_1Identity2gradients/MC/deconv_1/conv1d_transpose_grad/Conv2D=^gradients/MC/deconv_1/conv1d_transpose_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/MC/deconv_1/conv1d_transpose_grad/Conv2D
w
>gradients/MC/deconv_1/conv1d_transpose/ExpandDims_1_grad/ShapeConst*!
valueB"          *
dtype0
?
@gradients/MC/deconv_1/conv1d_transpose/ExpandDims_1_grad/ReshapeReshapeDgradients/MC/deconv_1/conv1d_transpose_grad/tuple/control_dependency>gradients/MC/deconv_1/conv1d_transpose/ExpandDims_1_grad/Shape*
T0*
Tshape0
~
<gradients/MC/deconv_1/conv1d_transpose/ExpandDims_grad/ShapeShapeMC/maxpool_1/MaxPool1d/Squeeze*
T0*
out_type0
?
>gradients/MC/deconv_1/conv1d_transpose/ExpandDims_grad/ReshapeReshapeFgradients/MC/deconv_1/conv1d_transpose_grad/tuple/control_dependency_1<gradients/MC/deconv_1/conv1d_transpose/ExpandDims_grad/Shape*
T0*
Tshape0
m
3gradients/MC/maxpool_1/MaxPool1d/Squeeze_grad/ShapeShapeMC/maxpool_1/MaxPool1d*
T0*
out_type0
?
5gradients/MC/maxpool_1/MaxPool1d/Squeeze_grad/ReshapeReshape>gradients/MC/deconv_1/conv1d_transpose/ExpandDims_grad/Reshape3gradients/MC/maxpool_1/MaxPool1d/Squeeze_grad/Shape*
T0*
Tshape0
?
1gradients/MC/maxpool_1/MaxPool1d_grad/MaxPoolGradMaxPoolGrad!MC/maxpool_1/MaxPool1d/ExpandDimsMC/maxpool_1/MaxPool1d5gradients/MC/maxpool_1/MaxPool1d/Squeeze_grad/Reshape*
ksize
*
paddingVALID*
T0*
strides
*
data_formatNHWC
d
6gradients/MC/maxpool_1/MaxPool1d/ExpandDims_grad/ShapeShape
MC/Reshape*
T0*
out_type0
?
8gradients/MC/maxpool_1/MaxPool1d/ExpandDims_grad/ReshapeReshape1gradients/MC/maxpool_1/MaxPool1d_grad/MaxPoolGrad6gradients/MC/maxpool_1/MaxPool1d/ExpandDims_grad/Shape*
T0*
Tshape0
P
gradients/MC/Reshape_grad/ShapeShapeMC/conv_1/add*
T0*
out_type0
?
!gradients/MC/Reshape_grad/ReshapeReshape8gradients/MC/maxpool_1/MaxPool1d/ExpandDims_grad/Reshapegradients/MC/Reshape_grad/Shape*
T0*
Tshape0
V
"gradients/MC/conv_1/add_grad/ShapeShapeMC/conv_1/Conv2D*
T0*
out_type0
`
$gradients/MC/conv_1/add_grad/Shape_1ShapeMC/conv_1/bias/bias/read*
T0*
out_type0
?
2gradients/MC/conv_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/MC/conv_1/add_grad/Shape$gradients/MC/conv_1/add_grad/Shape_1*
T0
?
 gradients/MC/conv_1/add_grad/SumSum!gradients/MC/Reshape_grad/Reshape2gradients/MC/conv_1/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
?
$gradients/MC/conv_1/add_grad/ReshapeReshape gradients/MC/conv_1/add_grad/Sum"gradients/MC/conv_1/add_grad/Shape*
T0*
Tshape0
?
"gradients/MC/conv_1/add_grad/Sum_1Sum!gradients/MC/Reshape_grad/Reshape4gradients/MC/conv_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
?
&gradients/MC/conv_1/add_grad/Reshape_1Reshape"gradients/MC/conv_1/add_grad/Sum_1$gradients/MC/conv_1/add_grad/Shape_1*
T0*
Tshape0
?
-gradients/MC/conv_1/add_grad/tuple/group_depsNoOp%^gradients/MC/conv_1/add_grad/Reshape'^gradients/MC/conv_1/add_grad/Reshape_1
?
5gradients/MC/conv_1/add_grad/tuple/control_dependencyIdentity$gradients/MC/conv_1/add_grad/Reshape.^gradients/MC/conv_1/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/MC/conv_1/add_grad/Reshape
?
7gradients/MC/conv_1/add_grad/tuple/control_dependency_1Identity&gradients/MC/conv_1/add_grad/Reshape_1.^gradients/MC/conv_1/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/MC/conv_1/add_grad/Reshape_1
{
&gradients/MC/conv_1/Conv2D_grad/ShapeNShapeN	transposeMC/conv_1/kernel/kernel/read*
T0*
out_type0*
N
?
3gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput&gradients/MC/conv_1/Conv2D_grad/ShapeNMC/conv_1/kernel/kernel/read5gradients/MC/conv_1/add_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations

?
4gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	transpose(gradients/MC/conv_1/Conv2D_grad/ShapeN:15gradients/MC/conv_1/add_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
	dilations

?
0gradients/MC/conv_1/Conv2D_grad/tuple/group_depsNoOp5^gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropFilter4^gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropInput
?
8gradients/MC/conv_1/Conv2D_grad/tuple/control_dependencyIdentity3gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropInput1^gradients/MC/conv_1/Conv2D_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropInput
?
:gradients/MC/conv_1/Conv2D_grad/tuple/control_dependency_1Identity4gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropFilter1^gradients/MC/conv_1/Conv2D_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/MC/conv_1/Conv2D_grad/Conv2DBackpropFilter"?