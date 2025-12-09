import pygame

# Cyberpunk / Retro Arcade Palette
COLOR_BG = (10, 10, 20)
COLOR_GRID = (40, 40, 60)
COLOR_NEON_GREEN = (57, 255, 20)
COLOR_NEON_PINK = (255, 20, 147)
COLOR_NEON_CYAN = (0, 255, 255)
COLOR_NEON_AMBER = (255, 191, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_RED = (255, 50, 50)
COLOR_DARK_RED = (100, 0, 0)

class RetroRenderer:
    def __init__(self, width, height, title="Tiny Mem Gym"):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.font = None
        self.title = title
        
    def _init_font(self):
        if not pygame.font.get_init():
            pygame.font.init()
        if self.font is None:
            # Try to find a retro font, fallback to default
            try:
                self.font = pygame.font.Font("freesansbold.ttf", 20)
            except:
                self.font = pygame.font.SysFont("Courier New", 20, bold=True)
                
    def clear(self):
        self.surface.fill(COLOR_BG)
        # Draw subtle grid
        for x in range(0, self.width, 40):
            pygame.draw.line(self.surface, COLOR_GRID, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 40):
            pygame.draw.line(self.surface, COLOR_GRID, (0, y), (self.width, y), 1)

    def draw_text(self, text, x, y, color=COLOR_WHITE, center=False, size=20):
        if self.font is None:
            self._init_font()
        
        # Simple shadow
        shadow_surf = self.font.render(text, True, (0, 0, 0))
        text_surf = self.font.render(text, True, color)
        
        if center:
            rect = text_surf.get_rect(center=(x, y))
            self.surface.blit(shadow_surf, (rect.x + 2, rect.y + 2))
            self.surface.blit(text_surf, rect)
        else:
            self.surface.blit(shadow_surf, (x + 2, y + 2))
            self.surface.blit(text_surf, (x, y))

    def draw_box(self, rect, color, filled=True, width=2):
        if filled:
            # Semi-transparent fill
            s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            s.fill((*color, 50))  # Low alpha
            self.surface.blit(s, (rect[0], rect[1]))
        
        pygame.draw.rect(self.surface, color, rect, width, border_radius=4)
        
        # Corner accents
        x, y, w, h = rect
        len_ = min(10, w//4)
        pygame.draw.line(self.surface, color, (x, y), (x + len_, y), width+1)
        pygame.draw.line(self.surface, color, (x, y), (x, y + len_), width+1)
        
        pygame.draw.line(self.surface, color, (x+w, y), (x+w-len_, y), width+1)
        pygame.draw.line(self.surface, color, (x+w, y), (x+w, y + len_), width+1)
        
        pygame.draw.line(self.surface, color, (x, y+h), (x+len_, y+h), width+1)
        pygame.draw.line(self.surface, color, (x, y+h), (x, y+h-len_), width+1)
        
        pygame.draw.line(self.surface, color, (x+w, y+h), (x+w-len_, y+h), width+1)
        pygame.draw.line(self.surface, color, (x+w, y+h), (x+w, y+h-len_), width+1)

    def draw_sidebar(self, x, width, instructions: list[str], controls: list[tuple[str, str]], level: int = 1):
        """Draw a sidebar with instructions and controls."""
        rect = (x, 0, width, self.height)
        
        # Background
        pygame.draw.rect(self.surface, (20, 20, 30), rect)
        pygame.draw.line(self.surface, COLOR_NEON_CYAN, (x, 0), (x, self.height), 2)
        
        pad = 20
        y = 20
        
        # Level Indicator
        self.draw_text(f"LEVEL {level}", x + width//2, y, COLOR_NEON_GREEN, center=True, size=30)
        y += 50
        
        # Title
        self.draw_text("MISSION", x + width//2, y, COLOR_NEON_AMBER, center=True, size=24)
        y += 40
        
        # Instructions
        for line in instructions:
            self.draw_text(line, x + pad, y, COLOR_WHITE, size=16)
            y += 25
            
        y += 20
        pygame.draw.line(self.surface, COLOR_GRID, (x + pad, y), (x + width - pad, y), 1)
        y += 20
        
        # Controls
        self.draw_text("CONTROLS", x + width//2, y, COLOR_NEON_PINK, center=True, size=24)
        y += 40
        
        for key_name, action in controls:
            # Draw Key Box
            key_w = 80
            key_h = 30
            key_rect = (x + pad, y, key_w, key_h)
            
            # Check if key is currently pressed? We'd need to pass that state.
            # For now, just static help.
            self.draw_box(key_rect, COLOR_NEON_CYAN, filled=False, width=1)
            self.draw_text(key_name, x + pad + key_w//2, y + 6, COLOR_NEON_CYAN, center=True, size=16)
            
            # Draw Action
            self.draw_text(action, x + pad + key_w + 10, y + 6, COLOR_WHITE, size=16)
            
            y += 40

    def get_frame(self):
        # Convert to numpy array (H, W, 3)
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    def draw_scanlines(self):
        # Draw scanlines
        for y in range(0, self.height, 2):
            pygame.draw.line(self.surface, (0, 0, 0, 50), (0, y), (self.width, y), 1)
