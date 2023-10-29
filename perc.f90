program perceptron




end program perceptron

! **********************************************************
! sigmoid activation function
! Returns the derivative of sigmoid if is_derivative is .true.
subroutine sigmoid(x, is_derivative)
    implicit none
    real, intent(inout) :: x 
    logical, intent(in) :: is_derivative

    if (is_derivative) then
        x = (1 / (1 + exp(-x))) * (1 - (1 / (1 + exp(-x))))
    else
        x = (1 / (1 + exp(-x)))
    end if
end subroutine sigmoid 
! ********************************************************** 
! tanh activation function
subroutine tanhaf(x, is_derivative)
    implicit none 
    real, intent(inout) :: xx 
    logical, intent(in) :: is_derivative 

    if (is_derivative) then
        x = 1 - (tanh(x))**2
    else 
        x = tanh(x)
    end if
end subroutine tanha
! **********************************************************
