class BaseAttackServer:
    def __call__(self, server):
        return self.change_functionality(server)

    def change_functionality(self, server):
        return server
