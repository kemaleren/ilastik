from abc import ABCMeta, abstractmethod

def _has_attribute( cls, attr ):
    return True if any(attr in B.__dict__ for B in cls.__mro__) else False

def _has_attributes( cls, attrs ):
    return True if all(_has_attribute(cls, a) for a in attrs) else False

class AppletGuiInterface():
    """
    This is the abstract interface to which all applet GUI classes should adhere.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def centralWidget( self ):
        """
        Abstract method.  Return the widget that will be displayed in the main viewer area.
        """
        raise NotImplementedError

    @abstractmethod
    def appletDrawers(self):
        """
        Abstract method.  Return a list of (drawer title, drawer widget) pairs for this applet.
        """
        raise NotImplementedError
    
    @abstractmethod
    def menus( self ):
        """
        Abstract method.  Return a list of QMenu widgets to be shown in the menu bar when this applet is visible.
        """
        raise NotImplementedError

    @abstractmethod
    def viewerControlWidget(self):
        """
        Abstract method.
        Return the widget that controls how the content of the central widget is displayed.
        Typically this consists of a layer list control.
        """
        raise NotImplementedError
    
    @abstractmethod
    def setImageIndex(self, imageIndex):
        """
        Abstract method.
        Called by the shell when the user has switched the input image he wants to view.
        The GUI should respond by updating the content of the central widget.
        """
        raise NotImplementedError

    @abstractmethod    
    def reset(self):
        """
        Abstract method.
        Called by the shell when the current project has been unloaded.
        The GUI should reset itself to its initial state, whatever that is.
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AppletGuiInterface:
            requiredMethods = [ 'centralWidget',
                                'appletDrawers',
                                'menus',
                                'viewerControlWidget',
                                'setImageIndex' ]
            return True if _has_attributes(C, requiredMethods) else False
        return NotImplemented

if __name__ == "__main__":
    class CustomGui(object):
        def centralWidget( self ):
            """
            Return the widget that will be displayed in the main viewer area.
            """
            raise NotImplementedError
    
        def appletDrawers(self):
            """
            Return a list of drawer widgets for this applet.
            """
            raise NotImplementedError
        
        def menus( self ):
            """
            Return a list of QMenu widgets to be shown in the menu bar when this applet is visible.
            """
            raise NotImplementedError
    
        def viewerControlWidget(self):
            """
            Return the widget that controls how the content of the central widget is displayed.
            Typically this consists of a layer list control.
            """
            raise NotImplementedError
        
        def setImageIndex(self, imageIndex):
            """
            Called by the shell when the user has switched the input image he wants to view.
            The GUI should respond by updating the content of the central widget.
            """
            raise NotImplementedError


    cg = CustomGui()
    assert issubclass( type(cg), AppletGuiInterface )
    assert isinstance( cg, AppletGuiInterface )













