class BaseAnonymiser:
    """
    Takes frame + faces list -> returns anonymised frame.
    """
    def apply(self, frame, faces):
        raise NotImplementedError
