from abc import abstractproperty
from ilastik.utility.subclassRegistry import SubclassRegistryMeta

class WorkflowMeta(SubclassRegistryMeta):
    pass

class Workflow( object ):
    __metaclass__ = WorkflowMeta # Provides Workflow.all_subclasses member

    @abstractproperty
    def applets(self):
        return []

    @abstractproperty
    def imageNameListSlot(self):
        return None

    @property
    def finalOutputSlot(self):
        return None

    @classmethod
    def getSubclass(cls, name):
        for subcls in cls.all_subclasses:
            if subcls.__name__ == name:
                return subcls
        raise RuntimeError("No known workflow class has name " + name)

