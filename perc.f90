program perceptron




end program perceptron

! **********************************************************
! sigmoid activation function
! Returns the derivative of sigmoid if is_derivative is .true.
subroutine sigmoid(x, is_derivative)
    implicit none
    real, intent(inout) :: x 
    logical, intent(in) :: is_derivative
    real :: sig 

    sig = 1 / (1 + exp(-x))

    if (is_derivative) then
        x = sig * (1 - sig)
    else
        x = sig
    end if
end subroutine sigmoid 
! ********************************************************** 
! tanh activation function
subroutine tanhaf(x, is_derivative)
    implicit none 
    real, intent(inout) :: xx 
    logical, intent(in) :: is_derivative 
    real :: t

    t = tanh(x)

    if (is_derivative) then
        x = 1 - t**2
    else 
        x = t
    end if
end subroutine tanhaf
! **********************************************************
