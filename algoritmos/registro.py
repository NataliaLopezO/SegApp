import SimpleITK as sitk
import nibabel as nib

def registrar(fixed_image_path, moving_image_path, fix_seg_path, moving_seg_path, path):
  fixed_image = sitk.ReadImage(fixed_image_path)
  moving_image = sitk.ReadImage(moving_image_path)
  
  fix_seg_image = sitk.ReadImage(fix_seg_path)
  mov_seg_image = sitk.ReadImage(moving_seg_path)

  fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
  moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)


  registration_method = sitk.ImageRegistrationMethod()
  registration_method.SetMetricAsMattesMutualInformation()
  registration_method.SetInterpolator(sitk.sitkLinear)
  registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.EachIteration)

  initial_transform = sitk.Transform()
  registration_method.SetInitialTransform(initial_transform)

  registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
  registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
  registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

  final_transform = registration_method.Execute(fixed_image, moving_image)

  registered_image = sitk.Resample(mov_seg_image, fix_seg_image, final_transform , sitk.sitkNearestNeighbor, 0.0, fixed_image.GetPixelID())
  register_array = sitk.GetArrayFromImage(registered_image)

  sitk.WriteImage(registered_image, path)
  # Cargar la imagen registrada utilizando nibabel
  registered_nifti = nib.load(path)
  registered_data = registered_nifti.get_fdata()

  return path